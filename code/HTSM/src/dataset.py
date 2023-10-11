import os
import json
import torch
import numpy as np
from PIL import Image
from skimage.io import imread
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from image_preprocessing import ImagePadder
from logger_utils import load_dict_from_json

class M4DSAROilSpillDataset(Dataset):
    def __init__(self, dir_data, list_images, which_set="train", file_stats_json="image_stats.json"):
        self.dir_data = dir_data
        self.which_set = which_set  # 数据集类型
        self.file_stats_json = file_stats_json
        try:
            self.dict_stats = load_dict_from_json(self.file_stats_json)
        except:
            dir_json = os.path.dirname(os.path.realpath(__file__))
            self.dict_stats = load_dict_from_json(os.path.join(dir_json, self.file_stats_json))
        
        # 设置图像和标签的目录路径
        self._dir_images = os.path.join(self.dir_data, "images")
        self._dir_labels = os.path.join(self.dir_data, "labels_1D")

        # 对图像和标签进行排序
        self._list_images = sorted(list_images)
        self._list_labels = [f.replace(".jpg", ".png") for f in self._list_images]

        # 创建一个ImagePadder对象（根据给定的目录路径）来填充图像
        dir_pad_image = os.path.dirname(self._dir_images)
        self._image_padder = ImagePadder(os.path.join("/".join(os.path.normpath(dir_pad_image).split(os.sep)[:-1]), "train", "images"))
        self._affine_transform = None

        self._image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    self.dict_stats["mean"],
                    self.dict_stats["mean"],
                    self.dict_stats["mean"]
                ],
                std=[
                    self.dict_stats["std"],
                    self.dict_stats["std"],
                    self.dict_stats["std"]
                ]
            ),
        ])

        if self.which_set == "train":
            self._affine_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])


    def __len__(self):
        return len(self._list_images)

    def __getitem__(self, idx):
        # 根据索引idx获取对应的图像和标签文件路径
        file_image = os.path.join(self._dir_images, self._list_images[idx])
        file_label = os.path.join(self._dir_labels, self._list_labels[idx])

        image = imread(file_image)
        label = imread(file_label)

        # 对图像和标签进行填充操作
        image = self._image_padder.pad_image(image)
        label = self._image_padder.pad_label(label)

        if self.which_set == "train":
            # 将图像和标签转换为张量
            image_tensor = torch.from_numpy(image)
            # H x W x 3
            label_tensor = torch.from_numpy(label)
            # H x W
            label_tensor = torch.unsqueeze(label_tensor, dim=-1)
            # H x W x 1

            # 通过拼接操作创建一个包含图像和标签的堆叠张量stacked
            stacked = torch.cat([image_tensor, label_tensor], dim=-1)
            # H x W x 4
            stacked = torch.permute(stacked, [2, 0, 1]) # 对堆叠张量进行维度变换
            # 4 x H x W
            stacked_transformed = self._affine_transform(stacked)   # 对堆叠张量进行仿射变换
            # 4 x H x W
            stacked_transformed = torch.permute(stacked_transformed, [1, 2, 0])
            # H x W x 4
            stacked_arr = stacked_transformed.numpy()

            image = stacked_arr[:, :, :-1]  # 将stacked_arr数组的第三个维度之前的所有元素切片
            # H x W x 3
            label = stacked_arr[:, :, -1]   # 将stacked_arr数组的最后一个通道的数据切片给label变量
            # H x W

        image = self._image_transform(image)
        return image, label

def get_dataloaders_for_training(dir_dataset, batch_size, random_state=None, num_workers=4):
    # dir_dataset = /content/Oil-Spill-Detection/Oil-Spill-Detection-Dataset

    # 从指定目录下的训练集文件夹中获取所有以".jpg"结尾的图像文件，并将它们按照文件名排序存储在list_images列表中
    list_images = sorted(
        [f for f in os.listdir(os.path.join(dir_dataset, "train", "images")) if f.endswith(".jpg")]
    )

    # 将list_images列表中的图像文件随机分割成训练集和验证集
    list_train_images, list_valid_images = train_test_split(
        list_images, test_size=0.05, shuffle=True, random_state=random_state,
    )
    print("dataset information")
    print(f"number of train samples: {len(list_train_images)}")
    print(f"number of validation samples: {len(list_valid_images)}")

    train_dataset = M4DSAROilSpillDataset(
        os.path.join(dir_dataset, "train"),
        list_train_images,
        which_set="train"
    )
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers # 并行加载数据的工作进程数
    )

    valid_dataset = M4DSAROilSpillDataset(
        os.path.join(dir_dataset, "train"),
        list_valid_images,
        which_set="valid"
    )
    valid_dataset_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_dataset_loader, valid_dataset_loader

def get_dataloader_for_inference(dir_dataset, batch_size=1, num_workers=4):
    list_inference_images = sorted(
        [f for f in os.listdir(os.path.join(dir_dataset, "test", "images")) if f.endswith(".jpg")]
    )

    inference_dataset = M4DSAROilSpillDataset(
        os.path.join(dir_dataset, "test"),
        list_inference_images,
        which_set="test"
    )
    inference_dataset_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return inference_dataset_loader, list_inference_images
