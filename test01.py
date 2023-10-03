# https://blog.csdn.net/smallworldxyl/article/details/121570419
#
import  numpy as np
class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # 了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            # print("lp={}".format(lp))
            # print("lt={}".format(lt))
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
            # print("hist=")
            # print(self.hist)
    def evaluate(self):
        # print("self hist")
        # print(self.hist)
        PA = np.diag(self.hist).sum() / self.hist.sum()
        # sum(1)为按行求和 sum(0)是按列求和
        CPA = np.diag(self.hist) / self.hist.sum(axis=0)
        MPA = np.nanmean(CPA)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        # print("PA={}, CPA={}, MPA ={}, iu={}, mean_iu={}, fwavacc={}".format(PA, CPA,MPA, iu, mean_iu, fwavacc))
        return PA, CPA, MPA ,iu, mean_iu, fwavacc
# ，pred和target的shape为【batch_size,channels,...】,2D和3D

import torch.nn as nn



class cross_wdice_softmax(nn.Module):
    def __init__(self, n_classes):
        super( cross_wdice_softmax, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _structure_loss(self,pred, mask):
        wbce=0

        # pred = torch.sigmoid(pred)
        # inter = ((pred * mask) * weit).sum(dim=(2, 3))
        # union = ((pred + mask) * weit).sum(dim=(2, 3))
        # wiou = 1 - (inter + 1) / (union - inter + 1)
        smooth = 1e-9
        N = pred.size()[0]
        inputs = pred.view(N,-1)
        targets = mask.view(N,-1)
        intersection = (inputs * targets)
        dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)
        # print("dice")
        # print(dice.mean())
        wdice_loss=1-dice

        # return (wbce + wdice_loss).mean()
        return (wdice_loss).mean()
    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)#24 224 224--->24 9 224 224
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            structloss = self._structure_loss(inputs[:,i], target[:,i])
            class_wise_dice.append(1.0 - structloss .item())
            loss += structloss  * weight[i]
        return loss / self.n_classes

import torch.nn.functional as F
# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(FocalLoss, self).__init__()
#
#     def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
#         inputs = F.sigmoid(inputs)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         BCE_EXP = torch.exp(-BCE)
#         focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
#         return focal_loss



if __name__=='__main__':
    miou=IOUMetric(2)
    import torch
    pre = torch.tensor([[[[1, 0, 0],
                         [0, 0, 0],
                         [0, 1, 0]]],

                        [[[1, 1, 0],
                         [0, 0, 1],
                         [0, 1, 0]]]])


    true = torch.tensor([[[[1, 1, 0],
                          [0, 1, 0],
                          [0, 1, 1]]],

                         [[[1, 1, 0],
                          [0, 1, 0],
                          [0, 1, 1]]]])

    # pre = torch.tensor([[[1, 0, 0],
    #                      [0, 0, 0],
    #                      [0, 1, 0]]])
    #
    # true = torch.tensor([[[1, 1, 0],
    #                       [0, 1, 0],
    #                       [0, 1, 1]]])
    print("true shape", true.shape)  # true shape torch.Size([2, 1, 3, 3])
    print("pre shape", pre.shape)  # shape torch.Size([2, 1, 3, 3])
    # 2 3 3
    pre = pre.data.cpu().numpy()
    # 2 3 3
    miouVal = 0
    accVal = 0
    true = true.data.cpu().numpy()

    miou.add_batch(pre ,true)
    # print(miou.evaluate())
    miou.evaluate()
    # accVal += miou.Pixel_Accuracy()
    # miouVal += miou.Mean_Intersection_over_Union()
# print('acc and miou are {},{}'.format(miou.Pixel_Accuracy(),miou.Mean_Intersection_over_Union()))