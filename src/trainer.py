import torch
import numpy as np
from tqdm import tqdm

class Trainer():
    def __init__(self, model, trainloader, validloader, criterion, optimizer, lr_scheduler = None,
                 num_epoch = 20, early_stop = 5, device = None):
        
        self.model = model.to(device)
        self.trainloader = trainloader
        self.validloader = validloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epoch = num_epoch
        self.early_stop = early_stop
        
        self.device = device

        self.best_loss = np.inf
        self.state = {"tra_loss": [], "val_loss":[]}
    
    def _epoch(self, x, y, istraining = True):
        pred = self.model(x)
        loss = self.criterion(pred, y)
        if istraining:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, pred
    
    def save(self, savepath):
        torch.save(self.model.state_dict(), savepath)

    def train(self):
        self.model.train()
        running_loss =  0.0
        for x, y in tqdm(self.trainloader):
            x = x.to(self.device)
            y = y.to(self.device)
            loss, preds = self._epoch(x, y)
            running_loss += loss.item()
        return running_loss/len(self.trainloader)
    
    def eval(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(self.validloader):
                x = x.to(self.device)
                y = y.to(self.device)
                loss, preds = self._epoch(x, y, False)
                running_loss += loss.item()
            return running_loss/len(self.validloader)
    
    def run(self):
        n_count = 0
        for epoch in range(self.num_epoch):
            print(f"Epoch {epoch+1}")
            tra_loss = self.train()
            val_loss = self.eval()

            print(f"Train loss: {tra_loss}, Valid loss: {val_loss}") 
            self.state["tra_loss"].append(tra_loss)
            self.state["val_loss"].append(val_loss)

            if epoch+1 % 10 == 0:
                self.save(f"./run/epoch_{epoch+1}_model_loss_{val_loss:.4f}.pth")

            if val_loss < self.best_loss:
                n_count = 0
                self.best_loss = val_loss
                self.save("./run/best_model.pth")
            else:
                n_count += 1
            
            if n_count == self.early_stop:
                print(f"Epoch {epoch} early stop.")
                break

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        return None