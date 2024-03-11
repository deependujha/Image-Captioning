"""_summary_
The module contains the dataset class and the dataloader function for the image captioning model.

Usage:
```python
    train_dataloader, test_dataloader = get_train_test_dataloader(
        train_df, test_df, img_dir, batch_size
    )

    for images, captions in train_dataloader:
        print(images.shape, captions)

    for images, captions in test_dataloader:
        print(images.shape, captions)
```
"""


import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


train_transform = v2.Compose(
    [
        v2.PILToTensor(),
        v2.Resize(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

test_transform = v2.Compose(
    [
        v2.PILToTensor(),
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_dataframe, img_dir, transform=None, target_transform=None
    ):
        self.annotations_df = annotations_dataframe  # annotation dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        image_caption = self.annotations_df.iloc[idx, 2]

        img_path = os.path.join(self.img_dir, self.annotations_df.iloc[idx, 0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image_caption = self.target_transform(image_caption)
        return image, image_caption


def get_train_test_dataloader(train_df, test_df, img_dir, batch_size):
    train_dataset = CustomImageDataset(
        annotations_dataframe=train_df,
        img_dir=img_dir,
        transform=train_transform,
        target_transform=None,
    )

    test_dataset = CustomImageDataset(
        annotations_dataframe=test_df,
        img_dir=img_dir,
        transform=test_transform,
        target_transform=None,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, test_dataloader
