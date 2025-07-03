import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from models.resnet import ResNet

import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
checkpoint_path = "/data01/dl24dorgio/dla/lab1/results/checkpoint/ResNet_Large_best.pth"
num_classes = 10  

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

cifar100_train = datasets.CIFAR100(root="./data", train=True, transform=transform, download=True)
cifar100_test = datasets.CIFAR100(root="./data", train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=False)

model = ResNet(num_classes=num_classes, depths=[7, 7], channels=[16, 32])
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()


def extract_features(model, dataloader, device):

    features = []
    labels = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)

            x = model.input_adapter(images)
            x = model.stage1(x)
            x = model.stage2(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)  # (B, C)

            features.append(x.cpu())
            labels.append(targets)

    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def train_and_evaluate_svm(X_train, y_train, X_test, y_test):

    print("Training linear SVM")
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nBaseline accuracy on CIFAR-100 using ResNet features: {acc * 100:.2f}%")
    return acc


print("Extracting train features")
X_train, y_train = extract_features(model, train_loader, device)

print("Extracting test features")
X_test, y_test = extract_features(model, test_loader, device)

print("Training linear SVM")
clf_svm = make_pipeline(StandardScaler(), SVC(kernel="linear"))
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"\nBaseline accuracy on CIFAR-100 using ResNet features (SVM): {acc_svm * 100:.2f}%")
