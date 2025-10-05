import torch
from torchvision import models

def export(weights="best_model.pth"):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1,3,224,224)
    torch.onnx.export(model, dummy_input, "crack_classifier.onnx",
                      input_names=["input"], output_names=["output"],
                      dynamic_axes=None, opset_version=12)
    print("Exported to crack_classifier.onnx")

if __name__ == "__main__":
    export()
