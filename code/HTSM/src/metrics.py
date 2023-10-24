import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use in validation loop
# compute mean pixel accuracy
def compute_mean_pixel_acc(true_label, pred_label):

    # 函数首先检查真实标签和预测标签的形状是否相同，以及它们是否具有三个维度。
    if true_label.shape != pred_label.shape:
        print("true_label has dimension", true_label.shape, ", pred_label values have shape", pred_label.shape)
        return

    if true_label.dim() != 3:
        print("true_label has dim", true_label.dim(), ", Must be 3.")
        return


    acc_sum = 0
    for i in range(true_label.shape[0]):
        # 首先将真实标签和预测标签转换为NumPy数组，然后将它们的数据类型转换为32位整型。
        true_label_arr = true_label[i, :, :].clone().detach().cpu().numpy()
        pred_label_arr = pred_label[i, :, :].clone().detach().cpu().numpy()
        true_label_arr = true_label_arr.astype(np.int32)
        pred_label_arr = pred_label_arr.astype(np.int32)

        # 比较真实标签和预测标签是否相同，并计算相同的像素数。
        same = (true_label_arr == pred_label_arr).sum()

        a, b = true_label_arr.shape
        total = a*b

        # 将相同的像素数除以总像素数
        acc_sum += same / total

    # 计算平均像素准确率，即将累加的准确率除以图像数量
    mean_pixel_accuracy = acc_sum / true_label.shape[0]
    return mean_pixel_accuracy



# Use in validation loop 
# compute mean IOU
def compute_mean_IOU(true_label, pred_label, num_classes=5):
    iou_list = list()
    present_iou_list = list()

    pred_label = pred_label.view(-1)
    true_label = true_label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # in computation of IoU.
    for sem_class in range(num_classes):
        pred_label_inds = (pred_label == sem_class)
        target_inds = (true_label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float("nan")
        else:
            intersection_now = (pred_label_inds[target_inds]).long().sum().item()
            union_now = pred_label_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    present_iou_list = np.array(present_iou_list)
    return np.mean(present_iou_list)

# Use in inference loop
def compute_class_IOU(true_label, pred_label, num_classes=4):
    iou_list = list()
    present_iou_list = list()

    pred_label = pred_label.view(-1)
    true_label = true_label.view(-1)

    per_class_iou = np.zeros(num_classes)

    # Note: Following for loop goes from 0 to (num_classes-1)
    # in computation of IoU.
    for sem_class in range(num_classes):
        # 通过比较预测标签和真实标签来确定预测标签中属于当前类别的像素，以及真实标签中属于当前类别的像素
        pred_label_inds = (pred_label == sem_class)
        target_inds = (true_label == sem_class)
        
        if target_inds.long().sum().item() == 0:    # 如果真实标签中没有当前类别的像素，那么将IOU设置为NaN
            iou_now = float("nan")
        else:
            intersection_now = (pred_label_inds[target_inds]).long().sum().item()
            union_now = pred_label_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        per_class_iou[sem_class] = (iou_now)
    return per_class_iou
