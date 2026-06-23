import argparse

def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, 
                        help = "choose a model [unet, resunet, attunet, resunet-tuned]", default="attunet")
    parser.add_argument("--batch_size", default=16, type = int, help="number of total epochs to run")
    parser.add_argument("--epochs", default=200, type = int, help="number of total epochs to run")
    parser.add_argument("--early_stop", default=15, type = int, help="number of early stop step")
    parser.add_argument("--amp", default=True, type=bool, help="whether to enable mixed precision training mode")
    parser.add_argument("--device", default="auto", type = str, help="which device will be used")
    parser.add_argument("--ce_weight", default=0.2, type=float, help="CrossEntropy weight")
    parser.add_argument("--resume", default="", type = str, metavar="PATH", help="path to checkpoint (default: None)")
    parser.add_argument("--name", default="default", type = str, help = "experiment name")
    return parser.parse_args()

def deploy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, 
                        help = "choose a model [unet, resunet, attunet, resunet-tuned]", default="attunet")
    parser.add_argument("--batch_size", default=16, type = int, help="number of total epochs to run")
    parser.add_argument("--amp", default=True, type=bool, help="whether to enable mixed precision training mode")
    parser.add_argument("--device", default="auto", type = str, help="which device will be used")
    parser.add_argument("--weight", default="", type = str, metavar="PATH", help="path to checkpoint (default: None)")
    parser.add_argument("--name", default="default", type = str, help = "experiment name")

    parser.add_argument("-o", dest="output", default="./unet.onnx", type=str, metavar="PATH", help="onnx savepath")
    return parser.parse_args()