import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from snntorch.functional import ce_rate_loss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNN, CNNSNN

def train_epoch(model, device, loader, optimizer, criterion, is_snn=False):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if is_snn:
            spk_rec = model(data)
            loss = criterion(spk_rec, target)
        else:
            out = model(data)
            loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)

def test_epoch(model, device, loader, criterion, is_snn=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if is_snn:
                spk_rec = model(data)
                out = spk_rec.mean(0)
                loss = criterion(spk_rec, target)
            else:
                out = model(data)
                loss = criterion(out, target)
            total_loss += loss.item() * data.size(0)
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def raster_plot(spk_tensor, path):
    # spk_tensor: (T, num_neurons)
    spike_times = []
    for i in range(spk_tensor.size(1)):
        times = (spk_tensor[:, i] > 0).nonzero(as_tuple=False).squeeze().tolist()
        spike_times.append(times if isinstance(times, list) else [times])
    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times)
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.savefig(path)
    plt.close()

def generate_raster(model, device, loader, path):
    model.eval()
    data, _ = next(iter(loader))
    data = data.to(device)
    with torch.no_grad():
        spk_rec = model(data[:1])
    raster_plot(spk_rec.squeeze(1).cpu(), path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "hybrid"], default="cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--timesteps", type=int, default=20, help="time steps for hybrid model"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if args.model == "cnn":
        model = CNN().to(device)
        is_snn = False
    else:
        model = CNNSNN(T=args.timesteps).to(device)
        is_snn = True

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = ce_rate_loss if is_snn else nn.CrossEntropyLoss()

    os.makedirs("results", exist_ok=True)
    save_path = os.path.join(
        "results", "cnn_baseline.pt" if args.model == "cnn" else "hybrid.pt"
    )

    train_losses, test_losses, accs = [], [], []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, device, train_loader, optimizer, criterion, is_snn
        )
        test_loss, acc = test_epoch(model, device, test_loader, criterion, is_snn)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accs.append(acc)
        print(
            f"Epoch {epoch}: train loss {train_loss:.4f}, test loss {test_loss:.4f}, acc {acc*100:.2f}%"
        )
    metrics = {"test_loss": test_loss, "test_accuracy": acc}

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    metrics_path = os.path.join("results", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    epochs = range(1, args.epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, test_losses, label="test loss")
    plt.plot(epochs, [a * 100 for a in accs], label="test acc (%)")
    plt.xlabel("Epoch")
    plt.legend()
    curve_path = os.path.join("results", f"train_curve_{args.model}.png")
    plt.savefig(curve_path)
    plt.close()

    if is_snn:
        raster_path = os.path.join("results", "spike_raster.png")
        generate_raster(model, device, test_loader, raster_path)
        print(f"Spike raster saved to {raster_path}")

if __name__ == '__main__':
    main()
