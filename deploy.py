import torch
import torch.nn as nn
import onnx
from src.unet import UNet
from src.resunet import ResUnet
from src.attention import AttUnet
from src.resunet_tuned import AttUnetPretrained
from src.data import MAX_HEIGHT, MAX_WIDTH
from utils.argparse import deploy_args

def get_model(model: nn.Module, weight: str):
    model_state = torch.load(weight)
    model.load_state_dict(model_state["state_dict"])
    model = torch.jit.script(model)
    return model

def check_model(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model, full_check=True)

def export_onnx(model, savepath):
    model.eval()
    dummy_input = torch.randn([32, 3, MAX_HEIGHT, MAX_WIDTH])
    try:
        torch.onnx.export(model, dummy_input, f = savepath,
                        #   opset_version=12,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"},
                                        "output": {0: "batch_size"}},
                          verbose=False)
        check_model(savepath)
        return True
    except Exception as exp:
        print(exp)
        return False
    
def main(args):
    if args.model == 'unet':
        model = UNet(3, 21)
    elif args.model == "resunet":
        model = ResUnet(3, 21)
    elif args.model == "attunet":
        model = AttUnet(3, 21)
    elif args.model == "resunet-tuned":
        model = AttUnetPretrained(3, 21)
    
    model = get_model(model, args.weight)
    isexported = export_onnx(model, args.output)
    if isexported:
        print("Model output into onnx successfully.")
    else:
        raise ValueError("Model output into onnx failed.")

if __name__ == "__main__":
    args = deploy_args()
    main(args)