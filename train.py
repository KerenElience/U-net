import torch
from torch.utils.data import DataLoader
from src.unet import UNet
from src.resunet import ResUnet
from src.loss import CEDiceLoss
from src.data import UnetDataset
from src.trainer import Trainer
from utils.utils import ROOT
from utils.plot import plot_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

def main():
    traindataset = UnetDataset(ROOT/"data", "train")
    valdataset = UnetDataset(ROOT/"data", "val")
    trainloader = DataLoader(traindataset, batch_size = 16, num_workers=2, shuffle=True, pin_memory = True, persistent_workers=True, prefetch_factor=2)
    valloader = DataLoader(valdataset, batch_size = 32, num_workers=2, shuffle=False, pin_memory = True, persistent_workers=True, prefetch_factor=2)

    num_epoch = 200
    early_stop = 15
    on_amp = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = UNet(3, 21)
    model = ResUnet(3, 21)
    criterion = CEDiceLoss(num_classes=21, ce_weight=0.1)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=3e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2, weight_decay=3e-4, momentum=0.9)

    # lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch)
    trainer = Trainer(model, trainloader, valloader, criterion, optimizer, lr_scheduler, num_epoch, 
                      early_stop=early_stop, device = device, on_amp=on_amp)
    trainer.run()
    plot_loss(trainer.state["tra_loss"], trainer.state["val_loss"], ROOT/"Figure"/"ResUnet_loss.png")

if __name__ == "__main__":
    main()