import os
import cv2
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torchvision

class DFDataset(Dataset):
    def __init__(self, data_name="CelebA-HQ", cache_dir="/data/huangjie", text_dir=None, image_dir=None, 
                 split_way="long", sub_dataset= False):
        super().__init__()
        self.data_name = data_name
        self.sub_dataset = sub_dataset

        if data_name == "CelebA-HQ":
            from datasets import load_dataset
            self.image_size = 128
            dataset = load_dataset("saitsharipov/CelebA-HQ", split="train", cache_dir=cache_dir)
            self.dataset_list = dataset.with_transform(self._transform)
        elif data_name == "CIFAR-100":
            self.image_size = 32
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            train = torchvision.datasets.CIFAR100(root=cache_dir, train=True, download=True, transform=transform)
            test = torchvision.datasets.CIFAR100(root=cache_dir, train=False, download=True, transform=transform)
            self.dataset_list = ConcatDataset([train, test])
        elif data_name == "Flicker30k":
            self.image_size = 512
            if not (text_dir and image_dir):
                raise ValueError("text_dir and image_dir must be provided for Flicker30k")
            df = pd.read_csv(text_dir).dropna(subset=["caption"]).groupby("image")["caption"].apply(list).to_dict()
            self.dataset_list = {
                os.path.join(image_dir, img): (
                    max(captions, key=len) if split_way == "long" else
                    min(captions, key=len) if split_way == "small" else
                    random.choice(captions)
                ) for img, captions in df.items() if captions
            }
            self.dataset_keys = list(self.dataset_list.keys())

    def _transform(self, examples):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return {"images": [preprocess(img) for img in examples["image"]]}

    def _open_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scale = min(self.image_size / img.shape[1], self.image_size / img.shape[0])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        x_offset, y_offset = (self.image_size - new_size[0]) // 2, (self.image_size - new_size[1]) // 2
        padded[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = img

        return padded

    def __len__(self):
        if self.sub_dataset:
            return 100
            # return int(len(self.dataset_list)*0.1)
        return len(self.dataset_list)

    def __getitem__(self, idx):
        if self.data_name == "Flicker30k":
            return self._transform({"image": [self._open_image(self.dataset_keys[idx])]})["images"][0], self.dataset_list[self.dataset_keys[idx]]
        return self.dataset_list[idx]

if __name__ == "__main__":
    datasets = [
        DFDataset(),
        DFDataset(data_name="CIFAR-100"),
        DFDataset(data_name="Flicker30k", text_dir="/data/huangjie/flickr30k/captions.txt", 
                  image_dir="/data/huangjie/flickr30k/Images/", image_size=512)
    ]
    dataloaders = [DataLoader(ds, batch_size=32) for ds in datasets]
    for dl in dataloaders:
        for i, batch in enumerate(dl):
            if i == 0:
                if isinstance(batch, dict):
                    images = batch['images']
                    print(images.shape)
                else:
                    images, text = batch
                    print(images.shape, text, len(text))
                break