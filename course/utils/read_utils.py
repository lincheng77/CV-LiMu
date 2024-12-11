import os
import pandas as pd
import torch
import torchvision


def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    csv_data = pd.read_csv('../data/banana-detection/bananas_train/label.csv')

    # 设置索引列
    csv_data = csv_data.set_index('img_name')

    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        # 所有图像
        images.append(torchvision.io.read_image(os.path.join('../data/banana-detection/bananas_train/images', img_name)))

        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))

    # unsqueeze(1)： 添加一个维度
    # / 256：对标签进行归一化，通常图像的尺寸为 256x256，归一化后边界框的坐标值在 [0, 1] 之间。
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        # self.features 图像张量
        # self.labels 图像标签
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    # 根据索引获取单个样本
    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    # 返回数据集的大小
    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""

    # 加载训练集的 DataLoader
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)

    # 加载验证集的 DataLoader
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter



def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

