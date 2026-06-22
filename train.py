import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.unet import UNet
from src.resunet import ResUnet
from src.attention import AttUnet
from src.resunet_tuned import AttUnetPretrained
from src.loss import CEDiceLoss
from src.data import UnetDataset
from src.trainer import Trainer
from utils.utils import ROOT, calc_rare_sample_weight
from utils.plot import plot_loss
from utils.argparse import train_args
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


#see the utils `precent, weight = calc_classes_weight_by_pixel(traindataset)`
pixel_precent = [0.59427402, 0.01427115, 0.0106742 , 0.01355912, 0.00916274,
                 0.00697495, 0.01796363, 0.02626262, 0.04365999, 0.01511205,
                 0.00756764, 0.01194504, 0.04175405, 0.01301906, 0.01596141,
                 0.10060095, 0.00747321, 0.00843392, 0.01269623, 0.01853402,
                 0.0101    ]
weight = torch.tensor([0.1       , 0.95004097, 1.27015189, 0.99992625, 1.47964967,
                       1.94369527, 0.75476786, 0.51627017, 0.31055463, 0.89717952,
                       1.79148634, 1.13503095, 0.3247301 , 1.0414023 , 0.84944074,
                       0.13477992, 1.81412038, 1.60749811, 1.06788016, 0.73154082,
                       1.34235462])

def main(args):
    batch_size = args.batch_size
    traindataset = UnetDataset(ROOT/"data", "train")
    valdataset = UnetDataset(ROOT/"data", "val")
    if args.model == "unet":
        rare_weight = 3.0
    else:
        rare_weight = 2.0
    sampler_weight = calc_rare_sample_weight(traindataset, pixel_precent, rare_weight = rare_weight, n_jobs=4)
    sampler = WeightedRandomSampler(sampler_weight, len(traindataset), replacement=True)
    # trainloader = DataLoader(traindataset, batch_size = batch_size, num_workers=2, shuffle=True, pin_memory = True, persistent_workers=True, prefetch_factor=2)
    trainloader = DataLoader(traindataset, batch_size = batch_size, sampler = sampler, num_workers=2, shuffle=False, pin_memory = True, persistent_workers=True, prefetch_factor=2)
    valloader = DataLoader(valdataset, batch_size = batch_size*2, num_workers=2, shuffle=False, pin_memory = True, persistent_workers=True, prefetch_factor=2)

    param_groups = []
    num_epoch = args.epochs
    early_stop = args.early_stop
    on_amp = args.amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if args.model == 'unet':
        model = UNet(3, 21)
        param_groups.append({"params": model.parameters(), "lr": 1e-2})
    elif args.model == "resunet":
        model = ResUnet(3, 21)
        param_groups.append({"params": model.parameters(), "lr": 1e-2})
    elif args.model == "attunet":
        model = AttUnet(3, 21)
        param_groups.append({"params": model.parameters(), "lr": 1e-2})
    elif args.model == "resunet-tuned":
        model = AttUnetPretrained(3, 21) # input_channel is not useful
        encoder_params =[params for params in model.encoder.parameters()]
        decoder_params =[params for name, params in model.named_parameters() if "encoder" not in name]
        param_groups.append({"params": encoder_params, "lr": 1e-4, "weight_decay": 1e-5}) 
        param_groups.append({"params": decoder_params, "lr": 1e-2, "weight_decay": 3e-4})
    criterion = CEDiceLoss(num_classes=21, weight=weight.to(device), ce_weight=args.ce_weight)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=3e-4)
    optimizer = torch.optim.SGD(param_groups, lr = 1e-2, weight_decay=3e-4, momentum=0.9)

    # lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch)
    if args.resume != "":
        model_state = torch.load(args.resume)
        model.load_state_dict(model_state["state_dict"])
        optimizer.load_state_dict(model_state["optimizer"])
        lr_scheduler.load_state_dict(model_state["lr_scheduler"])
    trainer = Trainer(model, trainloader, valloader, criterion, optimizer, lr_scheduler, num_epoch,
                      early_stop=early_stop, device = device, on_amp=on_amp)
    trainer.run()
    plot_loss(trainer.state["tra_loss"], trainer.state["val_loss"], ROOT/"Figure"/f"{args.model}_loss.png")

if __name__ == "__main__":
    args = train_args()
    main(args)