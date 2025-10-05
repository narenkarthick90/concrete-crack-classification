import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

def train():
    with open("models/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((cfg["imgsz"], cfg["imgsz"])),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]),
        "val": transforms.Compose([
            transforms.Resize((cfg["imgsz"], cfg["imgsz"])),
            transforms.ToTensor(),
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(cfg["data_dir"]+x, data_transforms[x]) for x in ["train", "val"]}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=cfg["batch"], shuffle=True) for x in ["train", "val"]}

    model = models.resnet18(pretrained=cfg["pretrained"])
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    best_acc = 0.0
    patience = 5  # stop if no improvement after 5 epochs
    counter = 0

    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

    for epoch in range(cfg["epochs"]):
        print(f"Epoch {epoch+1}/{cfg['epochs']}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "train":
                train_acc_list.append(epoch_acc.item())
                train_loss_list.append(epoch_loss)
            else:
                val_acc_list.append(epoch_acc.item())
                val_loss_list.append(epoch_loss)

            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    counter = 0
                    torch.save(model.state_dict(), "best_model.pth")
                    print(f"Improved! Model saved (val acc = {best_acc:.4f})")
                else:
                    counter += 1
                    print(f"No improvement for {counter}/{patience} epochs")

        if counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break

    # plt.plot(train_acc_list, label="Train Acc")
    # plt.plot(val_acc_list, label="Val Acc")
    # plt.legend()
    # plt.savefig("accuracy_plot.png")
    # print("Training complete. Best Val Acc:", best_acc.item())

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_acc_list, label="Train Acc", marker='o')
    plt.plot(val_acc_list, label="Val Acc", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_list, label="Train Loss", marker='o')
    plt.plot(val_loss_list, label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.tight_layout()

    plot_path = os.path.join("results", "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()  # prevents displaying the window

    print(f"Training curves saved at: {plot_path}")
    print(f"Training complete. Best Val Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()
