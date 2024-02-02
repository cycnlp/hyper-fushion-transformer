import os

import torch
from medpy import metric
from torch.utils.data import DataLoader
# from torchvision.transforms import transforms

import test01
from utils.dataset import DeeplabDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from config import BUSI_path
with open(os.path.join(BUSI_path, "val.txt"), "r") as f:
    val_lines = f.readlines()

import random

import numpy as np
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

val_img_ids=val_lines

val_dataset = DeeplabDataset(val_img_ids, BUSI_path, 'test')
val_gen = DataLoader(val_dataset, batch_size=1, num_workers=0, pin_memory=True, drop_last=True)

from mynets.hynet import net1

model = net1(2)
model_name = 'hynet'


path='/home/zmk/deeplearning/Dataset_BUSI_upup/result/model_best.pth'
model.load_state_dict(torch.load(path,map_location='cpu')['model'])


model=model.to(device)
model.eval()
import torch.nn.functional as F
i=0



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        # metric.binary.
        jc = metric.binary.jc(pred, gt)
        # return dice, hd95,jc
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        # return 1, 0,0
        return 1
    else:
        # return 0, 0,0
        return 0
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
mymiou=[]
myiu1=[]
myiu0=[]

mydes=[]
myse=[]
mysp=[]
myacc=[]
myf1=[]
mypa=[]


myacc_sklearn=[]
mypre_sklearn=[]
myrecall_sklearn=[]
myf1_sklearn=[]
myauc_sklearn=[]

metric_list = []
with torch.no_grad():



    # for x, (imgs, labels, edge) in enumerate(val_gen):
    for x, (imgs, labels,name) in enumerate(val_gen):

        print(name[0]+'png')
        imgs=imgs.to(device)
        labels=torch.squeeze(labels,dim=1)
        # outputs, out_edge = model(imgs)#outputs是输出的图像 (b c h w)
        # outputs,_,_ = model(imgs)#outputs是输出的图像 (b 2 h w)  每类对应一个通道，0是第一个通道，1是第二个通道#
        outputs= model(imgs)#outputs是输出的图像 (b 2 h w)  每类对应一个通道，0是第一个通道，1是第二个通道/
        outputs= outputs[0]
     

        outx = F.softmax(outputs, dim=1) # b c h w


        print(x)


        # ## 判断每个像素的类别
        # 取出最大值的通道就是類別 batchize c h w--->b h w
        pre= torch.argmax(outx,dim=1)#b  h w

        # for i in range(1,4):

        pre = pre.data.cpu().numpy()  # 1  h w
        labels = labels.data.cpu().numpy()


        mu = test01.IOUMetric(2)
        mu.add_batch(pre, labels)

        PA, CPA, MPA, iu, mean_iu, fwavacc = mu.evaluate()
        mymiou.append(mean_iu)
        myiu1.append(iu[1])
        myiu0.append(iu[0])


        for i in range(1,2):
        # for i in range(0,2):
         
            metric_list.append(calculate_metric_percase(pre.squeeze() == i, labels.squeeze() == i))


        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score

        labels=labels.flatten().squeeze()
        pre=pre.flatten().squeeze()

        acc=accuracy_score(labels.flatten(),pre.flatten())
        # pre=precision_score(labels.flatten(),pre.flatten())
        # print(np.unique(labels))
        # print(np.unique(pre))
        pre_score=precision_score(labels.flatten(),pre.flatten())
        recall=recall_score(labels.astype('int').flatten(),pre.flatten())
        f1=f1_score(labels.flatten(),pre.flatten())
        auc=roc_auc_score(labels.flatten(),pre.flatten())

        myacc_sklearn.append(acc)
        mypre_sklearn.append(pre_score)
        myrecall_sklearn.append(recall)
        myf1_sklearn.append(f1)
        myauc_sklearn.append(auc)


print(model_name)
print(path)
print("miou",np.mean(mymiou), np.std(mymiou,ddof=1))
print("iou1",np.mean(myiu1),np.std(myiu1,ddof=1))
print("iou0",np.mean(myiu0),np.std(myiu0,ddof=1))
#
print("这是sklearn的结果")
print("acc",np.mean(myacc_sklearn),np.std(myacc_sklearn,ddof=1))
print("precison",np.mean(mypre_sklearn),np.std(mypre_sklearn,ddof=1))
print("recall",np.mean(myrecall_sklearn),np.std(myrecall_sklearn,ddof=1))
print("f1",np.mean(myf1_sklearn),np.std(myf1_sklearn,ddof=1))
print("auc",np.mean(myauc_sklearn),np.std(myauc_sklearn,ddof=1))
print("des",np.mean(metric_list),np.std(metric_list,ddof=1))



