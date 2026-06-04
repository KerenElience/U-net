import torch
from torch.utils.data import DataLoader
from src.unet import UNet
from src.loss import BCEDiceLoss
from src.data import UnetDataset
from src.trainer import Trainer
from utils.utils import ROOT
from utils.plot import plot_loss
from tqdm import tqdm


def main():
    traindataset = UnetDataset(ROOT/"data", "train")
    valdataset = UnetDataset(ROOT/"data", "val")
    trainloader = DataLoader(traindataset, batch_size = 16, num_workers=2, shuffle=True, pin_memory = True)
    valloader = DataLoader(valdataset, batch_size = 32, num_workers=2, shuffle=False, pin_memory = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet(3, 3)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr = 1e-3, weight_decay=1e-4)
    trainer = Trainer(unet, trainloader, valloader, criterion, optimizer, None, 20, early_stop=5, device = device)
    trainer.run()
    plot_loss(trainer.state["tra_loss"], trainer.state["val_loss"], ROOT/"Figure"/"Unet_loss.png")

if __name__ == "__main__":
    main()