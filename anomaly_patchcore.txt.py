import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.neighbors import NearestNeighbors

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

data_root = "./mvtec_anomaly_detection"
category = "bottle"

batch_size = 32

save_dir = "./results_patchcore"
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


patch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

vis_transform = transforms.Compose([
    transforms.Resize((128, 128)),
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

train_dataset = MVTecDataset(train_paths, patch_transform)
test_dataset = MVTecDataset(test_paths, patch_transform)
vis_test_dataset = MVTecDataset(test_paths, vis_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
vis_test_loader = DataLoader(vis_test_dataset, batch_size=1, shuffle=False)

print("train:", len(train_dataset))
print("test:", len(test_dataset))


# =========================
# ResNet Feature Extractor
# =========================
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()


# =========================
# 정상 feature 추출
# =========================
normal_features = []

with torch.no_grad():
    for x, _ in train_loader:
        x = x.to(device)
        feat = resnet(x)
        normal_features.append(feat.cpu().numpy())

normal_features = np.concatenate(normal_features, axis=0)

print("normal feature shape:", normal_features.shape)


# =========================
# kNN
# =========================
knn = NearestNeighbors(n_neighbors=5)
knn.fit(normal_features)


# =========================
# Threshold 계산
# =========================
train_scores = []

for feat in normal_features:
    feat = feat.reshape(1, -1)
    dist, _ = knn.kneighbors(feat)
    score = dist.mean()
    train_scores.append(score)

threshold = np.percentile(train_scores, 95)
print("threshold:", threshold)


# =========================
# 테스트 + txt 저장
# =========================
txt_path = os.path.join(save_dir, "result_summary.txt")

correct_count = 0
total_count = 0

with open(txt_path, "w") as f:
    with torch.no_grad():
        for i, ((x, path), (x_vis, _)) in enumerate(zip(test_loader, vis_test_loader)):
            x = x.to(device)

            image_path = path[0]
            file_name = os.path.basename(image_path)
            defect_type = image_path.split(os.sep)[-2]

            ground_truth = "Normal" if defect_type == "good" else "Anomaly"

            feat = resnet(x).cpu().numpy()

            dist, _ = knn.kneighbors(feat)
            score = dist.mean()

            prediction = "Anomaly" if score > threshold else "Normal"
            correct = prediction == ground_truth

            if correct:
                correct_count += 1

            total_count += 1

            save_path = os.path.join(save_dir, f"patchcore_{i:03d}_{file_name}")

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            ax.imshow(x_vis.squeeze().permute(1, 2, 0))
            ax.set_title(f"{prediction}\nscore={score:.4f}")
            ax.axis("off")

            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

            # terminal 출력
            print("=" * 70)
            print("METHOD       : PatchCore-style")
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
            f.write("METHOD       : PatchCore-style\n")
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
    f.write("METHOD       : PatchCore-style\n")
    f.write(f"TOTAL        : {total_count}\n")
    f.write(f"CORRECT      : {correct_count}\n")
    f.write(f"ACCURACY     : {accuracy:.2f}%\n")
    f.write("=" * 70 + "\n")

print("=" * 70)
print("FINAL RESULT")
print("METHOD   : PatchCore-style")
print("TOTAL    :", total_count)
print("CORRECT  :", correct_count)
print("ACCURACY :", f"{accuracy:.2f}%")
print("TXT      :", txt_path)