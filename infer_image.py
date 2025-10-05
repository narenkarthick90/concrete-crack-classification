import torch
from torchvision import transforms, models
from PIL import Image

CLASSES = ["Negative", "Positive"]

def infer(weights="best_model.pth", image_path="example.jpg"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    print(f"Prediction: {CLASSES[pred.item()]}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to image file")
    parser.add_argument("--weights", type=str, default="best_model.pth", help="Path to trained model")
    args = parser.parse_args()

    infer(weights=args.weights, image_path=args.source)
