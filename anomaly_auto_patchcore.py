import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.neighbors import NearestNeighbors

# =========================
# 1. 설정
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

data_root = "./mvtec"
category = "bottle"

img_size = 128
batch_size = 32
epochs = 5

save_dir = "./compare_results"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 2. Dataset
# =========================
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

train_paths = glob.glob(os.path.join(data_root, category, "train", "good", "*.png"))
test_paths = glob.glob(os.path.join(data_root, category, "test", "*", "*.png"))

train_paths.sort()
test_paths.sort()

train_dataset = MVTecDataset(train_paths, transform)
test_dataset = MVTecDataset(test_paths, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("train:", len(train_dataset))
print("test:", len(test_dataset))

# =========================
# 3. Conv AutoEncoder
# =========================
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),     # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),    # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),   # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


ae_model = ConvAE().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

# =========================
# 4. AutoEncoder 학습
# =========================
for epoch in range(epochs):
    ae_model.train()
    total_loss = 0

    for x, _ in train_loader:
        x = x.to(device)

        recon = ae_model(x)
        loss = criterion(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[AE] Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")

# =========================
# 5. AE threshold 계산
# =========================
ae_model.eval()
ae_train_errors = []

with torch.no_grad():
    for x, _ in train_loader:
        x = x.to(device)
        recon = ae_model(x)
        errors = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
        ae_train_errors.extend(errors.cpu().numpy())

ae_threshold = np.percentile(ae_train_errors, 95)
print("AE threshold:", ae_threshold)

# =========================
# 6. PatchCore 스타일 feature extractor
# =========================
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 마지막 fc 제거
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()

# ImageNet normalization 필요
patchcore_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

patch_train_dataset = MVTecDataset(train_paths, patchcore_transform)
patch_test_dataset = MVTecDataset(test_paths, patchcore_transform)

patch_train_loader = DataLoader(patch_train_dataset, batch_size=32, shuffle=False)
patch_test_loader = DataLoader(patch_test_dataset, batch_size=1, shuffle=False)

# =========================
# 7. 정상 feature 추출
# =========================
normal_features = []

with torch.no_grad():
    for x, _ in patch_train_loader:
        x = x.to(device)
        feat = resnet(x)          # shape: [B, 512]
        normal_features.append(feat.cpu().numpy())

normal_features = np.concatenate(normal_features, axis=0)

print("normal feature shape:", normal_features.shape)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(normal_features)

# 정상 feature 기준 threshold
patch_train_scores = []

for feat in normal_features:
    feat = feat.reshape(1, -1)
    dist, _ = knn.kneighbors(feat)
    score = dist.mean()
    patch_train_scores.append(score)

patch_threshold = np.percentile(patch_train_scores, 95)
print("PatchCore-style threshold:", patch_threshold)

# =========================
# 8. 비교 테스트
# =========================
ae_model.eval()
resnet.eval()

# 시각화용 원본 transform
vis_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

vis_test_dataset = MVTecDataset(test_paths, vis_transform)
vis_test_loader = DataLoader(vis_test_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for i, ((x_vis, path), (x_patch, _)) in enumerate(zip(vis_test_loader, patch_test_loader)):
        image_path = path[0]
        file_name = os.path.basename(image_path)
        defect_type = image_path.split(os.sep)[-2]

        # -------------------------
        # AE prediction
        # -------------------------
        x_ae = x_vis.to(device)
        recon = ae_model(x_ae)

        ae_error = torch.mean((x_ae - recon) ** 2).item()
        ae_pred = "Anomaly" if ae_error > ae_threshold else "Normal"

        ae_map = torch.abs(x_ae - recon).cpu().squeeze().mean(dim=0).numpy()

        # -------------------------
        # PatchCore-style prediction
        # -------------------------
        x_patch = x_patch.to(device)
        feat = resnet(x_patch).cpu().numpy()

        dist, _ = knn.kneighbors(feat)
        patch_score = dist.mean()

        patch_pred = "Anomaly" if patch_score > patch_threshold else "Normal"

        # -------------------------
        # 저장용 시각화
        # -------------------------
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))

        ax[0].imshow(x_vis.squeeze().permute(1, 2, 0))
        ax[0].set_title(f"Input\n{defect_type}")
        ax[0].axis("off")

        ax[1].imshow(recon.cpu().squeeze().permute(1, 2, 0))
        ax[1].set_title("AE Reconstruction")
        ax[1].axis("off")

        ax[2].imshow(ae_map, cmap="hot")
        ax[2].set_title(f"AE Map\n{ae_pred}\nscore={ae_error:.4f}")
        ax[2].axis("off")

        ax[3].imshow(x_vis.squeeze().permute(1, 2, 0))
        ax[3].set_title(f"PatchCore-style\n{patch_pred}\nscore={patch_score:.4f}")
        ax[3].axis("off")

        save_path = os.path.join(save_dir, f"compare_{i:03d}_{file_name}")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print("=" * 50)
        print("file:", image_path)
        print("AE:", ae_pred, "score:", ae_error)
        print("PatchCore-style:", patch_pred, "score:", patch_score)
        print("saved:", save_path)

        if i == 20:
            break