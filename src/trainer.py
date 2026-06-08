import torch
import numpy as np
from tqdm import tqdm

class Trainer():
    def __init__(self, model, trainloader, validloader, criterion, optimizer, lr_scheduler = None,
                 num_epoch = 20, early_stop = 10, tolerance = 0.005, num_classes = 21, ignore_index = 255,
                 device = None, on_amp = False):
        
        self.model = model.to(device)
        self.trainloader = trainloader
        self.validloader = validloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epoch = num_epoch
        self.early_stop = early_stop
        self.tolerance = tolerance
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        self.device = device
        self.on_amp = on_amp
        if on_amp:
            self.scaler = torch.GradScaler(device)

        self.best_loss = np.inf
        self.best_miou = 0.0
        self.state = {"tra_loss": [], "val_loss":[], "lr": [], 
                      "miou": [], "class_iou": []}

    
    def _epoch(self, x, y, istraining = True):
        if self.on_amp:
            device_type = str(self.device).split(':')[0]
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                pred = self.model(x)
                loss = self.criterion(pred, y)
        else:
            pred = self.model(x)
            loss = self.criterion(pred, y)

        if istraining:
            self.optimizer.zero_grad()
            if self.on_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
        return loss, pred
    
    def save(self, savepath):
        torch.save(self.model.state_dict(), savepath)

    def train(self):
        self.model.train()
        running_loss =  0.0
        with tqdm(self.trainloader, desc = "Training") as pbar:
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                loss, preds = self._epoch(x, y)
                running_loss += loss.item()
                pbar.set_postfix(loss = f"{loss.item():.4f}")
        return running_loss/len(self.trainloader)
    
    @torch.no_grad()
    def eval(self):
        self.model.eval()
        running_loss = 0.0
        conf_mat = torch.zeros(self.num_classes, self.num_classes, device= self.device)
        with tqdm(self.validloader, desc="Validing") as pbar:
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                loss, preds = self._epoch(x, y, False)
                running_loss += loss.item()
                conf_mat += self.calc_conf_mat(preds, y, self.ignore_index)

                pbar.set_postfix(loss = f"{loss.item():.4f}")
            miou, class_iou = self.calc_miou(conf_mat)
            return running_loss/len(self.validloader), miou, class_iou
    
    def run(self):
        n_count = 0
        for epoch in range(self.num_epoch):
            print(f"Epoch {epoch+1}")
            tra_loss = self.train()
            val_loss, miou, class_iou = self.eval()

            print(f"Train loss: {tra_loss}, Valid loss: {val_loss}") 
            self.state["tra_loss"].append(tra_loss)
            self.state["val_loss"].append(val_loss)
            self.state["miou"].append(miou)
            self.state["class_iou"].append(class_iou)

            if self.lr_scheduler is not None:
                if hasattr(self.lr_scheduler, "reduce_on_plateau"):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.state["lr"].append(current_lr)

            if (epoch+1) % 10 == 0:
                self.save(f"./run/epoch_{epoch+1}_model_loss_{val_loss:.4f}.pth")

            # if val_loss < self.best_loss + self.tolerance:
            if miou > self.best_miou + self.tolerance:
                n_count = 0
                # self.best_loss = val_loss
                self.best_miou = miou
                self.save(f"./run/best_model.pth")
            else:
                n_count += 1
            
            if n_count == self.early_stop:
                print(f"Epoch {epoch} early stop.")
                break
        print(f"Training finished. Best mIoU = {self.best_miou:.4f}")
        return None
    
    @staticmethod
    @torch.no_grad()
    def calc_conf_mat(pred, target, ignore_index):
        num_classes = pred.shape[1]
        pred = pred.argmax(dim = 1)
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        if ignore_index is not None:
            mask = (target != ignore_index)
            pred = pred[mask]
            target = target[mask]
        
        idx = target*num_classes + pred
        count = torch.bincount(idx, minlength=num_classes*num_classes)
        return count.reshape(num_classes, num_classes)
    
    @staticmethod
    def calc_miou(conf_mat):
        tp = conf_mat.diag()
        fp = conf_mat.sum(0) - tp
        fn = conf_mat.sum(1) - tp

        iou = tp / (tp + fp + fn)
        miou = torch.nanmean(iou).item()
        return miou, iou.cpu().numpy()
