import torch
from torch.utils.data import DataLoader
from src.unet import UNet
from src.loss import CEDiceLoss
from src.data import UnetDataset
from src.trainer import Trainer
from utils.utils import ROOT
from utils.plot import plot_loss
from transformers import get_cosine_schedule_with_warmup

def main():
    traindataset = UnetDataset(ROOT/"data", "train")
    valdataset = UnetDataset(ROOT/"data", "val")
    trainloader = DataLoader(traindataset, batch_size = 32, num_workers=2, shuffle=True, pin_memory = True, persistent_workers=True, prefetch_factor=2)
    valloader = DataLoader(valdataset, batch_size = 32, num_workers=2, shuffle=False, pin_memory = True, persistent_workers=True, prefetch_factor=2)

    num_epoch = 100
    early_stop = 15
    on_amp = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet(3, 21)
    criterion = CEDiceLoss(num_classes=21, ignore_index=255)
    optimizer = torch.optim.Adam(unet.parameters(), lr = 1e-4, weight_decay=3e-5)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps= int(num_epoch*0.2),
        num_training_steps=len(trainloader)*num_epoch
    )
    trainer = Trainer(unet, trainloader, valloader, criterion, optimizer, lr_scheduler, num_epoch, 
                      early_stop=early_stop, device = device, on_amp=on_amp)
    trainer.run()
    plot_loss(trainer.state["tra_loss"], trainer.state["val_loss"], ROOT/"Figure"/"Unet_loss.png")

if __name__ == "__main__":
    main()