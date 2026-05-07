# synthetic.py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np
from PIL import Image
from config import IMG_SIZE, BATCH_SIZE, NUM_CLASSES

class SyntheticCheXpert(Dataset):
    def __init__(self, num_samples=2000, seed=None):
        self.num_samples = num_samples
        if seed is not None:
            np.random.seed(seed)
        self.data = np.random.randint(0, 255, (num_samples, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        self.labels = np.random.randint(0, 2, (num_samples, NUM_CLASSES))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx], mode='L').convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img)
        label = self.labels[idx].astype(np.float32)
        return img, label

def get_synthetic_client_loader(client_id):
    full = SyntheticCheXpert(num_samples=2000, seed=42)
    np.random.seed(client_id)
    main_class = client_id % NUM_CLASSES
    indices = []
    for i in range(len(full)):
        if full.labels[i][main_class] == 1:
            if np.random.rand() < 0.8:
                indices.append(i)
        else:
            if np.random.rand() < 0.2:
                indices.append(i)
    subset = Subset(full, indices[:500])
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

def get_synthetic_test_loader():
    full = SyntheticCheXpert(num_samples=500, seed=999)
    return DataLoader(full, batch_size=BATCH_SIZE, shuffle=False)