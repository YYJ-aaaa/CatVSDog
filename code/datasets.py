import torch.utils.data as data
import os
from glob import glob
import numpy as np
import cv2
from PIL import Image


class DogCat(data.Dataset):
    def __init__(self, data_dir, img_size=(224, 224), train=True,
                 transform=None, target_transform=None):
        self.img_size = img_size
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        # 读取图片文件列表
        self.imgs_path = glob('%s/*.jpg'% data_dir)
        self.imgs_label = []

        # 将文件名上的标签读取出来 转化为0 1
        dict = {'dog': 1, 'cat': 0}
        for img_path in self.imgs_path:
            img_name = os.path.basename(os.path.splitext(img_path)[0])
            label_name = os.path.splitext(img_name)[0]
            #数据集只存在dog 和 cat的情况
            self.imgs_label.append(np.array(dict[label_name]))


    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = cv2.imread(img_path)
        target = self.imgs_label[index]
        img = cv2.resize(img, self.img_size)

        # img = img.astype('float32') / 255.0
        # img = np.transpose(img, (2, 0, 1))
        # img = torch.from_numpy(img)
        #target = torch.from_numpy(target)
        #img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
            return len(self.imgs_path)