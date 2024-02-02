import os
import time
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm



from utils.utils_other import cross_wdice_softmax


os.environ['CUDA_ENABLE_DEVICES'] = '0'

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 记录每一轮训练的的lpss
epoch_train_loss=[]
# 记录每一轮训练的的miou
epoch_train_miou=[]


from torch.nn import CrossEntropyLoss
loss_Cross= CrossEntropyLoss()

loss_dice=cross_wdice_softmax(2)

def fit_one_epoch(net,w1,w2,w3,epoch,gen):

        print("len(gen)",len(gen))
        net = net.train()
        train_miou=0
        iu0=0
        iu1=0
        train_total_loss = 0
        train_dice=0
        for i, (imgs,labels) in enumerate(gen):
            imgs = imgs
            # 2 1 256 256
            labels = labels.squeeze(1)
            # 2  256 256
            # print("这是第多少个batchsize  ", i)
            if torch.cuda.is_available():
                # print("\n")
                # 2 3 256 256
                imgs=imgs.to(device=device)
                # 2 1 256 256
                # 2  256 256
                labels=labels.to(device=device)
            # 2 2 256 256
            # print("imgs.shape ", imgs.shape)
            outputs=net(imgs)

            croloss1 = loss_Cross(outputs[0], labels.long())
            croloss2 = loss_Cross(outputs[1], labels.long())
            croloss3 = loss_Cross(outputs[2], labels.long())
            #
            lossx1 = loss_dice(outputs[0], labels)
            lossx2 = loss_dice(outputs[1], labels)
            lossx3 = loss_dice(outputs[2], labels)
            lossx = w1* (croloss1 + lossx1) + w2* (croloss2 + lossx2) + w3 * (lossx3 + croloss3)

            optimizer.zero_grad()
            lossx.backward()
            optimizer.step()


def  val_one_epoch(net, w1,w2,w3, epoch, val_gen):
    print("val")
    net.eval()

    total_eval_loss = 0

    with torch.no_grad():
        for i, (val_imgs, val_labels) in enumerate(val_gen):
            val_labels = val_labels.squeeze(1)

            if (torch.cuda.is_available()):
                val_imgs = val_imgs.to(device=device)
                val_labels = val_labels.long().to(device=device)
            # print(val_imgs.shape)
            # print(val_labels.shape)
            outputs = net(val_imgs)
            croloss1 = loss_Cross(outputs[0], val_labels.long())
            eval_loss = loss_dice(outputs[0], val_labels)+croloss1
            # outputs = outputs[0]

            # total_eval_loss = total_eval_loss +0.8* eval_loss.item()+0.2*dece_loss.item()
            total_eval_loss = total_eval_loss + eval_loss.item()
    return total_eval_loss / len(val_gen)


def createmode():

    # for i in range(5):
    #   1.unet 2015
    global model, model_name
    from mynets.hynet import net1
    model = net1(2)
    model_name = 'hynet'


    model = model.to(device=device)

    return model,model_name

if __name__ == "__main__":

        import random

        import numpy as np
        seed = 2021
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        NUM_CLASSES = 2
        w1,w2,w3=1,1,1

        from config import BUSI_path

        # ---------------------------#
            #   读取数据集对应的txt
            # ---------------------------#
        with open(os.path.join(BUSI_path, "train.txt"), "r") as f:
            train_lines = f.readlines()
        with open(os.path.join(BUSI_path, "val.txt"), "r") as f:
            val_lines = f.readlines()
        train_img_ids = train_lines
        val_img_ids = val_lines

        model,model_name = createmode()

        # print(model)
        print("model------------name-----")
        print(model_name)
        print('model')
        print(model)

        savepath='/home/zmk/deeplearning/Dataset_BUSI_upup/result'

        path = savepath + model_name + '/'


        # exit(0)

        lr = 1e-2  # 将刚开始学习率设置大一点
        Init_Epoch = 0
        max_epoch = 100
        Batch_size = 12

        # params = model.parameters()
        # optimizer = torch.optim.Adam(params, 0.0001, betas=(0.5, 0.999))
        #
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        iterator = tqdm(range(max_epoch), ncols=70)

        start_epoch=0

        from utils.dataset import DeeplabDataset

        train_dataset   = DeeplabDataset(train_img_ids,BUSI_path, 'train')
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,     drop_last=True)


        val_dataset   = DeeplabDataset(val_img_ids,  BUSI_path,'val')
        val_gen = DataLoader(val_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,     drop_last=True)


        print("train length   ", len(train_img_ids))
        print("val length   ", len(val_img_ids))
     

        # epoch_size      = len(train_lines) // Batch_size
        epoch_size      = len(train_img_ids) // Batch_size

        if epoch_size == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        min_loss = 10000
        maxmiou = -1
        # for epoch in range(start_epoch,args.num_epochs ):
        for epoch in iterator:
        # for epoch in range(2):
            start = time.time()
            print("epoch======{}".format(epoch))
            fit_one_epoch(model,w1,w2,w3, epoch, gen)
            # print(w1, w2, w3)
            val_loss=   val_one_epoch(model,w1,w2,w3, epoch,val_gen)

            lr_ = lr * (1.0 - epoch / max_epoch) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            end = time.time()


            print(" one epoch  need time s ",end-start)
            save_file = {"model": model.state_dict() }

            if (epoch == 0):
                if (os.path.exists(path)):
                    print(path)
                    raise "path 存在"

            if  val_loss < min_loss:
                min_loss = val_loss
                print("save model")
                # 保存模型语句
                # torch.save(model.state_dict(), "model.pth")
                torch.save(save_file, path + "/model_best.pth")

            if (epoch == max_epoch - 1):
               torch.save(save_file, path + "/model_{}.pth".format(epoch))

        torch.cuda.empty_cache()
        del  model

