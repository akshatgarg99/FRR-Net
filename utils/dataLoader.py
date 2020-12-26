import os
import collections
import torch
import numpy as np
from torch.utils import data
import cv2


class camvidLoader(data.Dataset):
    def __init__(
            self,
            root,
            split="train",
            is_transform=False,
            img_norm=True,
            test_mode=False,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.CAMVID_MEAN = [0.41189489566336, 0.4251328133025, 0.4326707089857]
        self.CAMVID_STD = [0.27413549931506, 0.28506257482912, 0.28284674400252]
        self.n_classes = 12
        self.files = collections.defaultdict(list)

        if not self.test_mode:
            for split in ["train", "test", "val"]:
                file_list = os.listdir(root + "/" + split)
                self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
            img_name = self.files[self.split][index]
            img_path = self.root + "/" + self.split + "/" + img_name
            lbl_path = self.root + "/" + self.split + "annot/" + img_name

        img = cv2.imread(img_path)
        lbl = cv2.imread(lbl_path)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        # img = cv2.resize(img, (self.img_size[0], self.img_size[1]),interpolation = cv2.INTER_NEAREST)  # uint8 with RGB mode
        lbl = lbl[4:-4, :, 0]
        img = img[4:-4]

        #augmentation
        gen = random.random()
        if gen >0.2:
            img = img[::-1,:,:].copy()
            lbl=lbl[::-1,:].copy()

        elif 0.2 >= gen > 0.4:
            img = img[:,::-1,:].copy()
            lbl=lbl[:,::-1].copy()

        img = img[:, :, ::-1]  # RGB -> BGR

        img = img.astype(np.float32)

        img = img / 255.0
        img -= self.CAMVID_MEAN
        img = img / self.CAMVID_STD
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def decode_segmap(self, temp, plot=False):
        Sky = [128, 128, 128]
        Building = [128, 0, 0]
        Pole = [192, 192, 128]
        Road = [128, 64, 128]
        Pavement = [60, 40, 222]
        Tree = [128, 128, 0]
        SignSymbol = [192, 128, 128]
        Fence = [64, 64, 128]
        Car = [64, 0, 128]
        Pedestrian = [64, 64, 0]
        Bicyclist = [0, 128, 192]
        Unlabelled = [0, 0, 0]

        label_colours = np.array(
            [
                Sky,
                Building,
                Pole,
                Road,
                Pavement,
                Tree,
                SignSymbol,
                Fence,
                Car,
                Pedestrian,
                Bicyclist,
                Unlabelled,
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


