import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

data_root = "./mvtec_anomaly_detection"
category = "bottle"

img_size = 128
batch_size = 32
epochs = 5

save_dir = "./results_autoencoder"
os.makedirs(save_dir, exist_ok=True)


class MVTecDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, path


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

train_paths = glob.glob(
    os.path.join(data_root, category, "train", "good", "*.png")
)

test_paths = glob.glob(
    os.path.join(data_root, category, "test", "*", "*.png")
)

train_paths.sort()
test_paths.sort()

train_dataset = MVTecDataset(train_paths, transform)
test_dataset = MVTecDataset(test_paths, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("train:", len(train_dataset))
print("test:", len(test_dataset))


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, 3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64, 3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32, 3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 3, 3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


model = ConvAutoEncoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 학습
# =========================
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, _ in train_loader:
        x = x.to(device)

        recon = model(x)
        loss = criterion(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")


# =========================
# Threshold 계산
# =========================
model.eval()
train_errors = []

with torch.no_grad():
    for x, _ in train_loader:
        x = x.to(device)
        recon = model(x)

        errors = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
        train_errors.extend(errors.cpu().numpy())

threshold = np.percentile(train_errors, 95)
print("threshold:", threshold)


# =========================
# 테스트 + txt 저장
# =========================
txt_path = os.path.join(save_dir, "result_summary.txt")

correct_count = 0
total_count = 0

with open(txt_path, "w") as f:
    model.eval()

    with torch.no_grad():
        for i, (x, path) in enumerate(test_loader):
            x = x.to(device)

            image_path = path[0]
            file_name = os.path.basename(image_path)
            defect_type = image_path.split(os.sep)[-2]

            ground_truth = "Normal" if defect_type == "good" else "Anomaly"

            recon = model(x)

            score = torch.mean((x - recon) ** 2).item()
            prediction = "Anomaly" if score > threshold else "Normal"
            correct = prediction == ground_truth

            if correct:
                correct_count += 1

            total_count += 1

            error_map = torch.abs(x - recon).cpu().squeeze().mean(dim=0)

            save_path = os.path.join(save_dir, f"ae_{i:03d}_{file_name}")

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))

            ax[0].imshow(x.cpu().squeeze().permute(1, 2, 0))
            ax[0].set_title("Input")
            ax[0].axis("off")

            ax[1].imshow(recon.cpu().squeeze().permute(1, 2, 0))
            ax[1].set_title("Reconstruction")
            ax[1].axis("off")

            ax[2].imshow(error_map, cmap="hot")
            ax[2].set_title(f"{prediction}\nscore={score:.4f}")
            ax[2].axis("off")

            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

            # terminal 출력
            print("=" * 70)
            print("METHOD       : AutoEncoder")
            print("FILE         :", file_name)
            print("DEFECT TYPE  :", defect_type)
            print("GROUND TRUTH :", ground_truth)
            print("PREDICTION   :", prediction)
            print("SCORE        :", f"{score:.6f}")
            print("THRESHOLD    :", f"{threshold:.6f}")
            print("CORRECT      :", correct)
            print("SAVED IMAGE  :", save_path)

            # txt 저장
            f.write("=" * 70 + "\n")
            f.write("METHOD       : AutoEncoder\n")
            f.write(f"FILE         : {file_name}\n")
            f.write(f"DEFECT TYPE  : {defect_type}\n")
            f.write(f"GROUND TRUTH : {ground_truth}\n")
            f.write(f"PREDICTION   : {prediction}\n")
            f.write(f"SCORE        : {score:.6f}\n")
            f.write(f"THRESHOLD    : {threshold:.6f}\n")
            f.write(f"CORRECT      : {correct}\n")
            f.write(f"SAVED IMAGE  : {save_path}\n")
            f.write("=" * 70 + "\n\n")

    accuracy = correct_count / total_count * 100

    f.write("\n")
    f.write("=" * 70 + "\n")
    f.write("FINAL RESULT\n")
    f.write("=" * 70 + "\n")
    f.write("METHOD       : AutoEncoder\n")
    f.write(f"TOTAL        : {total_count}\n")
    f.write(f"CORRECT      : {correct_count}\n")
    f.write(f"ACCURACY     : {accuracy:.2f}%\n")
    f.write("=" * 70 + "\n")

print("=" * 70)
print("FINAL RESULT")
print("METHOD   : AutoEncoder")
print("TOTAL    :", total_count)
print("CORRECT  :", correct_count)
print("ACCURACY :", f"{accuracy:.2f}%")
print("TXT      :", txt_path)