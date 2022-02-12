import numpy as np
import pandas as pd
import rasterio
import torch
import torch.utils.data as td
from torch.utils.data import Dataset
from torchvision import transforms

means = np.array([660.91006849, 697.52952431, 442.62290759, 3228.94035111,
                  1148.48317061, 2566.46554959, 3117.52790816, 3410.80595118,
                  2166.43746621, 1264.30793458, 332.8328504, 3394.26638567])

stds = np.array([503.07559619, 321.38592137, 259.49743129, 883.19662534,
                 500.21024954, 629.1578845, 838.47806702, 875.89228695,
                 746.00491461, 641.0788046, 180.62920835, 781.54216725])


class SatelliteData(Dataset):

    def __init__(self, csv_file, transform=None, use_channels="all", train=True):
        """
        Args:
            csv_file: Path to csv file
            transform: Optional transforms
            use_channels: "all" or "rgb"
            train: Train or test split
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.use_channels = use_channels

        train_val_split = int(0.9 * len(self.df))
        self.train_df = self.df.iloc[:train_val_split]
        self.test_df = self.df.iloc[train_val_split:]

        self.df = self.train_df if train else self.test_df
        print(f"Dataset loaded. Data points: {self.df.shape[0]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, label_path = self.df.iloc[idx]
        image = rasterio.open(image_path).read()
        label = rasterio.open(label_path).read()
        image, label = image.astype("float32"), label.astype("float32")
        # Cleaning labels to be one of [0,1,2]
        label[label > 2] = 2
        if self.use_channels == "rgb":
            bgr_image = image[2:5, :]
            # rgb_image = bgr_image[..., ::-1].copy()
            image = bgr_image
            image = transforms.Normalize(means[2:5], stds[2:5])(torch.tensor(image))
        else:
            image = transforms.Normalize(means, stds)(torch.tensor(image))
        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    dataset = SatelliteData('data/data.csv', train=True, use_channels='rgb')
    dataloader = td.DataLoader(dataset, batch_size=1, shuffle=True)
    for data in dataloader:
        print(data['image'].shape)
