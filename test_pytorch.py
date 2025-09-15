import os
from torch.utils.data import Dataset
from PIL import Image


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
import functools

torch.load = functools.partial(torch.load, weights_only=False)
# Transform (ImageNet preprocessing)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class ImageNetSampleImages(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # list all JPEGs
        self.files = [f for f in os.listdir(root) if f.endswith(".JPEG")]

        # build synset-to-index mapping
        self.synsets = sorted({fname.split("_")[0] for fname in self.files})
        self.synset_to_idx = {s: i for i, s in enumerate(self.synsets)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        synset = fname.split("_")[0]
        label = self.synset_to_idx[synset]
        path = os.path.join(self.root, fname)

        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    

valdir = "./imagenet-sample-images"
dataset = ImageNetSampleImages(valdir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

# Load model
model = fbnet("fbnet_a", pretrained=True).eval()

# Evaluate
def evaluate(model, loader):
    correct1, correct5, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, top5 = outputs.topk(5, 1, True, True)
            labels = labels.view(-1, 1)

            correct1 += (top5[:, 0:1] == labels).sum().item()
            correct5 += (top5 == labels).sum().item()
            total += labels.size(0)
    return 100.0*correct1/total, 100.0*correct5/total

top1, top5 = evaluate(model, loader)
print(f"Subset accuracy on {len(dataset)} samples:")
print(f"Top-1: {top1:.2f}%   Top-5: {top5:.2f}%")