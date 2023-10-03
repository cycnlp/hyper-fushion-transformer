import os

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
import torch
import albumentations as A
import cv2

class DeeplabDataset(Dataset):
    # def __init__(self, train_lines, dataset_path, mode=None, transform=None, target_transform=None):
    def __init__(self, train_lines, dataset_path, mode=None):
        super(DeeplabDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.mode = mode

        self.dataset_path = dataset_path
        # self.transform = transform
        # self.target_transform = target_transform

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        # self.gt_transform = transforms.Compose([
        #     transforms.ToTensor()])
        # self.gt_transform=torch.as_tensor()

        self.transform = A.Compose(
            [


                # 平移缩放旋转
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),

                # 0.5水平翻转
                # 围绕Y轴水平翻转   垂直翻转
                A.HorizontalFlip(),
                # 围绕X轴垂直翻转  水平翻转
                A.VerticalFlip(),
                # 输入随机旋转90度，零次或多次。随机旋转90度
                A.RandomRotate90()
            ]
        )

        self.val_transforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return self.train_batches

    # def __getitem__(self, index):
    def __getitem__(self, index):

        # print('查看mask的元素值 只能有01 2 3')
        # print(set(label1.data.cpu().numpy().flatten()))

        if (self.mode == 'train'):
            # global img_x
            annotation_line = self.train_lines[index]
            name = annotation_line.split('\n')[0]

            # print("name  " + name)
            # image=cv2.imread(os.path.join(os.path.join(self.dataset_path, "train","image"), name + ".png"))
            path1=os.path.join(os.path.join(self.dataset_path, "images"), name + ".png")
            # print("path1 " + path1)

            image=cv2.imread(path1)

            # print("image\n")
            # print(image.shape)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.resize(image, (256, 256))
            image = cv2.resize(image, (224, 224))
            # path2=os.path.join(os.path.join(self.dataset_path,  "mask"), name + "_mask.png")

            mask = cv2.imread(os.path.join(os.path.join(self.dataset_path,  "mask"), name + "_mask.png"), 0)
            # print(set(mask.flatten()))
            # for i in range(mask.shape[0]):
            #     for j in range(mask.shape[1]):
            #         if (mask[i][j] == 255):
            #             mask[i][j] = 1
            # print("resize前")
            # print(set(mask.flatten()))
            # mask1 = cv2.resize(mask, (256, 256))

            # mask = cv2.resize(mask, (256, 256),interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (224, 224),interpolation=cv2.INTER_NEAREST)

            # print(np.unique(mask2==mask1))

            # print("resize后")
            # print(set(mask.flatten()))
            transformed = self.transform(image=image, mask=mask)
            image = self.img_transform(transformed['image'])# tensor 3 h w
            # print("68行")
            # zz=transformed['mask']#256 256
            # print(set(zz.flatten()))#0 1 2 3
            # gt = self.gt_transform(transformed['mask'])  #tensor 1 h w
            import torch
            # gt = torch.unsqueeze(torch.as_tensor(transformed['mask']),0)  #tensor h w
            gt = torch.unsqueeze(torch.as_tensor(transformed['mask']/255),0)  #tensor h w


            # print("----gt--",gt.unique())
            # print("mask元素")
            # print(set(gt.squeeze().numpy().flatten()))

            return image, gt
        if (self.mode == 'val'):

            annotation_line = self.train_lines[index]
            name = annotation_line.split('\n')[0]
            # 从文件中读取图像
            # image1 = Image.open(os.path.join(os.path.join(self.dataset_path, 'train', "image"), name + ".png")).convert('RGB')
            # mask1 = Image.open(os.path.join(os.path.join(self.dataset_path, 'train', "mask"), name + ".png"))

            image = cv2.imread(os.path.join(os.path.join(self.dataset_path, "images"), name + ".png"))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

            mask = cv2.imread(os.path.join(os.path.join(self.dataset_path,   "mask"), name + "_mask.png"), 0)

            # mask = cv2.resize(mask, (256, 256),interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (224, 224),interpolation=cv2.INTER_NEAREST)

            # print("68行")
            # zz=transformed['mask']#256 256
            # print(set(zz.flatten()))#0 1 2 3
            # gt = self.gt_transform(transformed['mask'])  #tensor 1 h w
            import torch
            import torch
            # gt = torch.unsqueeze(torch.as_tensor(mask), dim=0)
            gt = torch.unsqueeze(torch.as_tensor(mask)/255, dim=0)
            # print("72行")
            # print(set(gt.squeeze().numpy().flatten()))

            img=self.val_transforms(image)



            return img, gt
        if(self.mode=='test'):
            annotation_line = self.train_lines[index]
            # name = annotation_line.split()[0]
            name = annotation_line.split('\n')[0]
            # 从文件中读取图像
            # image1 = Image.open(os.path.join(os.path.join(self.dataset_path, 'train', "image"), name + ".png")).convert('RGB')
            # mask1 = Image.open(os.path.join(os.path.join(self.dataset_path, 'train', "mask"), name + ".png"))

            image = cv2.imread(os.path.join(os.path.join(self.dataset_path, "images"), name + ".png"))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

            # rgb_img=image/255

            mask = cv2.imread(os.path.join(os.path.join(self.dataset_path, "mask"), name + "_mask.png"), 0)
            # print(set(mask.flatten()))
            # for i in range(mask.shape[0]):
            #     for j in range(mask.shape[1]):
            #         if (mask[i][j] == 255):
            #             mask[i][j] = 1
            # print("resize前")
            # print(set(mask.flatten()))

            # mask = cv2.resize(mask, (256, 256))
            # mask = cv2.resize(mask, (256, 256))
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

            # print("68行")
            # zz=transformed['mask']#256 256
            # print(set(zz.flatten()))#0 1 2 3
            # gt = self.gt_transform(transformed['mask'])  #tensor 1 h w
            import torch
            import torch
            gt = torch.unsqueeze(torch.as_tensor(mask)/255, dim=0)
            # print("72行")
            # print(set(gt.squeeze().numpy().flatten()))

            img = self.val_transforms(image)

            # print("68行")
            # zz=transformed['mask']#256 256
            # print(set(zz.flatten()))#0 1 2 3
            # gt = self.gt_transform(transformed['mask'])  #tensor 1 h w


            # return img, gt,name,image
            return img, gt,name


if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize([256, 256],interpolation=0),
        transforms.Resize([256, 256], interpolation=transforms.InterpolationMode.NEAREST),
        # transforms.Resize([224, 224],interpolation=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # mask只需要转换为tensor
    val_transforms = transforms.Compose([

        transforms.ToTensor(),
        # transforms.as
        # transforms.Resize([256, 256],interpolation=0),
        transforms.Resize([256, 256], transforms.InterpolationMode.NEAREST)
        # transforms.Resize([224, 224],interpolation=0)
    ])

    from train_val_test_config import heart_path
    import torch



    # # ---------------------------#
    # #   读取数据集对应的txt
    # # ---------------------------#
    # with open(os.path.join(heart_path, "train.txt"), "r") as f:
    #     train_lines = f.readlines()
    # with open(os.path.join(heart_path, "val.txt"), "r") as f:
    #     val_lines = f.readlines()
    # train_img_ids = train_lines
    # val_img_ids=val_lines
    #
    # # train_dataset = DeeplabDataset(train_img_ids, heart_path, 'train')
    # train_dataset = DeeplabDataset( val_img_ids,heart_path, 'val')
    # # print(train_dataset[0])
    # yy = train_dataset[1]
    # print(yy)
