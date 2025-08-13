# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, matthews_corrcoef
# import matplotlib.pyplot as plt
# from unet3 import UNet3D
# import time
# from DS_TransUNet3D import UNet3D
# import configs as configs
# from datUtils import *
# from TransUnet import VisionTransformer

# def main(argv):
#     load_model(argv[0])
#     goFakeValidation(model)

# def loadData(n1, n2, n3, path, fname):
#     gx = np.fromfile(path + fname, dtype=np.float32)
#     gm, gs = np.mean(gx), np.std(gx)
#     gx = (gx - gm) / gs
#     gx = np.reshape(gx, (n3, n2, n1))
#     gx = np.transpose(gx)
#     return gx

# def loadData1(n1, n2, n3, path, fname):
#     lx = np.fromfile(path + fname, dtype=np.int8)
#     lm, ls = np.mean(lx), np.std(lx)
#     lx = lx - lm
#     lx = lx / ls
#     lx = np.reshape(lx, (n3, n2, n1))
#     lx = np.transpose(lx)
#     return lx

# def load_model(checkpoint_path):
#     global model
#     config = configs.get_r50_b16_config()
#     model = VisionTransformer(config, img_size=128, num_classes=21843, zero_head=False, vis=False)
#     checkpoint = torch.load("/root/autodl-tmp/Channel-checkTU/checkpointTUnet.43.pth")
#     model.load_state_dict(checkpoint['state_dict'])
#     model.eval()

# def calculate_metrics(pred, label):
#     """计算精确度、召回率、IOU、Dice 系数、Precision 和 F1 分数"""
#     tp = np.sum((pred == 1) & (label == 1))
#     tn = np.sum((pred == 0) & (label == 0))
#     fp = np.sum((pred == 1) & (label == 0))
#     fn = np.sum((pred == 0) & (label == 1))

#     accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
#     recall = tp / (tp + fn + 1e-10)
#     precision = tp / (tp + fp + 1e-10)
#     iou = tp / (tp + fp + fn + 1e-10)
#     dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
#     f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

#     # 计算MCC
#     mcc = matthews_corrcoef(label.flatten(), pred.flatten())

#     return accuracy, recall, iou, dice, precision, f1_score, mcc

# def create_trilinear_weights(input_size, overlap):
#     weights = np.ones((input_size, input_size, input_size))
#     for i in range(input_size):
#         for j in range(input_size):
#             for k in range(input_size):
#                 factor_i = 1.0
#                 factor_j = 1.0
#                 factor_k = 1.0
#                 if i < overlap:
#                     factor_i = (i + 1) / (overlap + 1)
#                 elif i >= input_size - overlap:
#                     factor_i = (input_size - i) / (overlap + 1)
#                 if j < overlap:
#                     factor_j = (j + 1) / (overlap + 1)
#                 elif j >= input_size - overlap:
#                     factor_j = (input_size - j) / (overlap + 1)
#                 if k < overlap:
#                     factor_k = (k + 1) / (overlap + 1)
#                 elif k >= input_size - overlap:
#                     factor_k = (input_size - k) / (overlap + 1)
#                 weights[i, j, k] = factor_i * factor_j * factor_k
#     return weights

# def goFakeValidation(model):
#     n1, n2, n3 = 256, 256, 256  # 原始数据大小
#     input_size = 128  # 模型训练使用的子图像大小
#     overlap = 0  # 重叠区域大小
#     seisPath = "/root/autodl-tmp/data/validationTransUnet/nx/"
#     lxpath = "/root/autodl-tmp/data/validationTransUnet/lx/"
#     predPath = "/root/autodl-tmp/data/validationTransUnet/px/"
#     ks = [251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]

#     trilinear_weights = create_trilinear_weights(input_size, overlap)

#     for k in ks:
#         start_time = time.time()  # 记录开始时间

#         fname = str(k)
#         gx = loadData(n1, n2, n3, seisPath, fname + '.dat')
#         gx = np.reshape(gx, (1, 1, n1, n2, n3))
#         gx = torch.from_numpy(gx).float()
#         gs = np.transpose(gx)
#         lx = loadData1(n1, n2, n3, lxpath, fname + '.dat')
#         lx = np.reshape(lx, (1, 1, n1, n2, n3))
#         lx = torch.from_numpy(lx).float()

#         m1, m2, m3 = input_size, input_size, input_size
#         step1 = m1 - overlap
#         step2 = m2 - overlap
#         step3 = m3 - overlap

#         num_chunks_n1 = (n1 - input_size) // step1 + 1
#         num_chunks_n2 = (n2 - input_size) // step2 + 1
#         num_chunks_n3 = (n3 - input_size) // step3 + 1

#         fx = np.zeros((n1, n2, n3), dtype=np.single)
#         weight = np.zeros((n1, n2, n3), dtype=np.single)
#         avg_accuracy, avg_recall, avg_iou, avg_dice, avg_precision, avg_f1_score, avg_mcc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         num_chunks = 0
#         chunk_metrics = []

#         with torch.no_grad():
#             for i in range(num_chunks_n1):
#                 for j in range(num_chunks_n2):
#                     for k in range(num_chunks_n3):
#                         start_n1 = i * step1
#                         end_n1 = start_n1 + input_size
#                         start_n2 = j * step2
#                         end_n2 = start_n2 + input_size
#                         start_n3 = k * step3
#                         end_n3 = start_n3 + input_size

#                         g = gx[:, :, start_n1:end_n1, start_n2:end_n2, start_n3:end_n3]
#                         l = lx[:, :, start_n1:end_n1, start_n2:end_n2, start_n3:end_n3]

#                         f = model(g)

#                         if isinstance(f, (tuple, list)):
#                             f = f[0]

#                         f = f.cpu().numpy()
#                         l = l.cpu().numpy()

#                         binary_fp = (f > 0.5).astype(int)
#                         binary_ls = (l > 0.5).astype(int)
#                         accuracy, recall, iou, dice, precision, f1_score, mcc = calculate_metrics(binary_fp, binary_ls)
#                         chunk_metrics.append((accuracy, recall, iou, dice, precision, f1_score))

#                         avg_accuracy += accuracy
#                         avg_recall += recall
#                         avg_iou += iou
#                         avg_dice += dice
#                         avg_precision += precision
#                         avg_f1_score += f1_score
#                         avg_mcc += mcc
#                         num_chunks += 1
#                         print(f"Metrics for chunk {num_chunks}: Accuracy={accuracy * 100:.2f}%, Recall={recall * 100:.2f}%, "
#                               f"IOU={iou * 100:.2f}%, Dice={dice * 100:.2f}%, Precision={precision * 100:.2f}%, "
#                               f"F1 Score={f1_score * 100:.2f}%, MCC={mcc:.4f}")

#                         fx[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += f[0, 0] * trilinear_weights
#                         weight[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += trilinear_weights

#             fx = np.divide(fx, weight, out=np.zeros_like(fx), where=weight != 0)
        
#         avg_accuracy /= num_chunks
#         avg_recall /= num_chunks
#         avg_iou /= num_chunks
#         avg_dice /= num_chunks
#         avg_precision /= num_chunks
#         avg_f1_score /= num_chunks
#         avg_mcc /= num_chunks

#         end_time = time.time()
#         elapsed_time = end_time - start_time  # 计算每个数据体的处理时间

#         print(f"Average metrics for {fname}: Accuracy={avg_accuracy * 100:.2f}%, Recall={avg_recall * 100:.2f}%, "
#               f"IOU={avg_iou * 100:.2f}%, Dice={avg_dice * 100:.2f}%, Precision={avg_precision * 100:.2f}%, "
#               f"F1 Score={avg_f1_score * 100:.2f}%, MCC={avg_mcc:.4f}, Time={elapsed_time:.2f}s")
        
#         plot_pr_curve(lx.flatten(), fx.flatten(), fname, avg_mcc)
#         plot_roc_curve(lx.flatten(), fx.flatten(), fname, avg_mcc)

# def plot_pr_curve(gx, fp, fname, avg_mcc):
#     gx_binary = np.clip(np.rint(gx), 0, 1)
#     precisions, recalls, thresholds = precision_recall_curve(gx_binary, fp, pos_label=1)
#     ap = average_precision_score(gx_binary, fp)

#     plt.figure()
#     plt.plot(recalls, precisions, label=f'AP={ap:.4f}, MCC={avg_mcc:.4f}')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend(loc='best')
#     plt.grid(True)
#     plt.savefig(f'PR curveTU_{fname}.png')
#     plt.show()

# def plot_roc_curve(gx, fp, fname, avg_mcc):
#     gx_binary = np.clip(np.rint(gx), 0, 1)
#     fpr, tpr, thresholds = roc_curve(gx_binary, fp, pos_label=1)
#     roc_auc = auc(fpr, tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}, MCC={avg_mcc:.4f}')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend(loc='best')
#     plt.grid(True)
#     plt.savefig(f'ROC curveTU_{fname}.png')
#     plt.show()

# if __name__ == "__main__":
#     main(["checkpointTUnet.43.pth"])

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, matthews_corrcoef
import matplotlib.pyplot as plt
from unet3 import UNet
import time
from DS_TransUNet3D import UNet3D
import configs as configs
from datUtils import *
from TransUnet import VisionTransformer

def main(argv):
    # 加载两个模型
    load_models(argv[0])
    goFakeValidation(model, ds_model,unet_model)  # 调用验证函数，并传递两个模型

def loadData(n1, n2, n3, path, fname):
    gx = np.fromfile(path + fname, dtype=np.float32)
    gm, gs = np.mean(gx), np.std(gx)
    gx = (gx - gm) / gs
    gx = np.reshape(gx, (n3, n2, n1))
    gx = np.transpose(gx)
    return gx

def loadData1(n1, n2, n3, path, fname):
    lx = np.fromfile(path + fname, dtype=np.int8)
    lm, ls = np.mean(lx), np.std(lx)
    lx = lx - lm
    lx = lx / ls
    lx = np.reshape(lx, (n3, n2, n1))
    lx = np.transpose(lx)
    return lx

def load_models(checkpoint_path):
    global model, ds_model, unet_model
    # Vision Transformer模型
    config = configs.get_r50_b16_config()
    model = VisionTransformer(config, img_size=128, num_classes=21843, zero_head=False, vis=False)
    checkpoint = torch.load("/root/autodl-tmp/Channel-checkTU/checkpointTUnet.43.pth")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # DS_TransUNet3D模型
    ds_model = UNet3D(128, 1) # 修改为实际输入输出通道
    ds_checkpoint =torch.load("/root/autodl-tmp/Channel-checkDS/checkpointDS.43.pth") # 使用实际路径
    ds_model.load_state_dict(ds_checkpoint['state_dict'])
    ds_model.eval()
    
    unet_model = UNet()  # Initialize your U-Net model, modify if needed
    unet_checkpoint = torch.load("/root/autodl-tmp/Channel-checkUNet1/checkpointUNet.43.pth")  # Use the correct path
    unet_model.load_state_dict(unet_checkpoint['state_dict'])
    unet_model.eval()
def calculate_metrics(pred, label):
    """计算精确度、召回率、IOU、Dice 系数、Precision 和 F1 分数"""
    tp = np.sum((pred == 1) & (label == 1))
    tn = np.sum((pred == 0) & (label == 0))
    fp = np.sum((pred == 1) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    # 计算MCC
    mcc = matthews_corrcoef(label.flatten(), pred.flatten())

    return accuracy, recall, iou, dice, precision, f1_score, mcc

def create_trilinear_weights(input_size, overlap):
    weights = np.ones((input_size, input_size, input_size))
    for i in range(input_size):
        for j in range(input_size):
            for k in range(input_size):
                factor_i = 1.0
                factor_j = 1.0
                factor_k = 1.0
                if i < overlap:
                    factor_i = (i + 1) / (overlap + 1)
                elif i >= input_size - overlap:
                    factor_i = (input_size - i) / (overlap + 1)
                if j < overlap:
                    factor_j = (j + 1) / (overlap + 1)
                elif j >= input_size - overlap:
                    factor_j = (input_size - j) / (overlap + 1)
                if k < overlap:
                    factor_k = (k + 1) / (overlap + 1)
                elif k >= input_size - overlap:
                    factor_k = (input_size - k) / (overlap + 1)
                weights[i, j, k] = factor_i * factor_j * factor_k
    return weights

def goFakeValidation(model, ds_model, unet_model):
    n1, n2, n3 = 256, 256, 256  # 原始数据大小
    input_size = 128  # 模型训练使用的子图像大小
    overlap = 0  # 重叠区域大小
    seisPath = "/root/autodl-tmp/data/validationTransUnet/nx/"
    lxpath = "/root/autodl-tmp/data/validationTransUnet/lx/"
    predPath = "/root/autodl-tmp/data/validationTransUnet/px/"
    ks = [251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]

    trilinear_weights = create_trilinear_weights(input_size, overlap)

    for k in ks:
        start_time = time.time()  # 记录开始时间

        fname = str(k)
        gx = loadData(n1, n2, n3, seisPath, fname + '.dat')
        gx = np.reshape(gx, (1, 1, n1, n2, n3))
        gx = torch.from_numpy(gx).float()
        gs = np.transpose(gx)
        lx = loadData1(n1, n2, n3, lxpath, fname + '.dat')
        lx = np.reshape(lx, (1, 1, n1, n2, n3))
        lx = torch.from_numpy(lx).float()

        m1, m2, m3 = input_size, input_size, input_size
        step1 = m1 - overlap
        step2 = m2 - overlap
        step3 = m3 - overlap

        num_chunks_n1 = (n1 - input_size) // step1 + 1
        num_chunks_n2 = (n2 - input_size) // step2 + 1
        num_chunks_n3 = (n3 - input_size) // step3 + 1

        fx_vit = np.zeros((n1, n2, n3), dtype=np.single)
        fx_ds = np.zeros((n1, n2, n3), dtype=np.single)
        fx_unet = np.zeros((n1, n2, n3), dtype=np.single)
        weight = np.zeros((n1, n2, n3), dtype=np.single)
        avg_accuracy, avg_recall, avg_iou, avg_dice, avg_precision, avg_f1_score, avg_mcc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        num_chunks = 0
        chunk_metrics = []

        with torch.no_grad():
            for i in range(num_chunks_n1):
                for j in range(num_chunks_n2):
                    for k in range(num_chunks_n3):
                        start_n1 = i * step1
                        end_n1 = start_n1 + input_size
                        start_n2 = j * step2
                        end_n2 = start_n2 + input_size
                        start_n3 = k * step3
                        end_n3 = start_n3 + input_size

                        g = gx[:, :, start_n1:end_n1, start_n2:end_n2, start_n3:end_n3]
                        l = lx[:, :, start_n1:end_n1, start_n2:end_n2, start_n3:end_n3]

                        # VisionTransformer模型预测
                        f_vit = model(g)
                        if isinstance(f_vit, (tuple, list)):
                            f_vit = f_vit[0]
                        f_vit = f_vit.cpu().numpy()

                        # DS_TransUNet3D模型预测
                        f_ds = ds_model(g)
                        if isinstance(f_ds, (tuple, list)):
                            f_ds = f_ds[0]
                        f_ds = f_ds.cpu().numpy()
                        
                        f_unet = unet_model(g)
                        if isinstance(f_unet, (tuple, list)):
                            f_unet = f_unet[0]
                        f_unet = f_unet.cpu().numpy()
                        
                        l = l.cpu().numpy()
                        # 计算指标
                        binary_fp_vit = (f_vit > 0.5).astype(int)
                        binary_fp_ds = (f_ds > 0.5).astype(int)
                        binary_fp_unet = (f_unet > 0.5).astype(int)
                        binary_ls = (l > 0.5).astype(int)
                        # binary_ls = (l > 0.5).long()
                        accuracy_vit, recall_vit, iou_vit, dice_vit, precision_vit, f1_score_vit, mcc_vit = calculate_metrics(binary_fp_vit, binary_ls)
                        accuracy_ds, recall_ds, iou_ds, dice_ds, precision_ds, f1_score_ds, mcc_ds = calculate_metrics(binary_fp_ds, binary_ls)
                        accuracy_unet, recall_unet, iou_unet, dice_unet, precision_unet, f1_score_unet, mcc_unet = calculate_metrics(binary_fp_unet, binary_ls)

                        avg_accuracy += (accuracy_vit + accuracy_ds) / 2
                        avg_recall += (recall_vit + recall_ds) / 2
                        avg_iou += (iou_vit + iou_ds) / 2
                        avg_dice += (dice_vit + dice_ds) / 2
                        avg_precision += (precision_vit + precision_ds) / 2
                        avg_f1_score += (f1_score_vit + f1_score_ds) / 2
                        avg_mcc += (mcc_vit + mcc_ds) / 2
                        num_chunks += 1

                        fx_vit[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += f_vit[0, 0] * trilinear_weights
                        fx_ds[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += f_ds[0, 0] * trilinear_weights
                        fx_unet[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += f_unet[0, 0] * trilinear_weights
                        weight[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += trilinear_weights

            fx_vit = np.divide(fx_vit, weight, out=np.zeros_like(fx_vit), where=weight != 0)
            fx_ds = np.divide(fx_ds, weight, out=np.zeros_like(fx_ds), where=weight != 0)
            fx_unet = np.divide(fx_unet, weight, out=np.zeros_like(fx_ds), where=weight != 0)
        avg_accuracy /= num_chunks
        avg_recall /= num_chunks
        avg_iou /= num_chunks
        avg_dice /= num_chunks
        avg_precision /= num_chunks
        avg_f1_score /= num_chunks
        avg_mcc /= num_chunks

        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算每个数据体的处理时间

        print(f"Average metrics for {fname}: Accuracy={avg_accuracy * 100:.2f}%, Recall={avg_recall * 100:.2f}%, "
              f"IOU={avg_iou * 100:.2f}%, Dice={avg_dice * 100:.2f}%, Precision={avg_precision * 100:.2f}%, "
              f"F1 Score={avg_f1_score * 100:.2f}%, MCC={avg_mcc:.4f}, Time={elapsed_time:.2f}s")
        
        # 绘制模型对比曲线
        plot_pr_curve(lx.flatten(), fx_vit.flatten(), fx_ds.flatten(), fx_unet.flatten(),fname, avg_mcc)
        plot_roc_curve(lx.flatten(), fx_vit.flatten(), fx_ds.flatten(),fx_unet.flatten(), fname, avg_mcc)
# 原始RGB值
rgb3 = (68, 114, 196)
# rgb1 = (232, 72, 132)
# rgb1 = (241, 141, 0)
rgb1 = (247, 171, 0)
# rgb2 = (9, 147, 150)
rgb2 = (43, 156, 161)
# 转换为0到1的范围
color1 = tuple([x / 255 for x in rgb1])
color2 = tuple([x / 255 for x in rgb2])
color3 = tuple([x / 255 for x in rgb3])
def plot_pr_curve(gx, fp_vit, fp_ds, fp_unet,fname, avg_mcc):
    gx_binary = np.clip(np.rint(gx), 0, 1)
    precisions_vit, recalls_vit, thresholds_vit = precision_recall_curve(gx_binary, fp_vit, pos_label=1)
    ap_vit = average_precision_score(gx_binary, fp_vit)
    precisions_ds, recalls_ds, thresholds_ds = precision_recall_curve(gx_binary, fp_ds, pos_label=1)
    ap_ds = average_precision_score(gx_binary, fp_ds)
    precisions_unet, recalls_unet, thresholds_unet = precision_recall_curve(gx_binary, fp_unet, pos_label=1)
    ap_unet = average_precision_score(gx_binary, fp_unet)
    plt.figure()
    plt.plot(recalls_ds, precisions_ds, label=f'DS-TransUnet AP={ap_ds:.4f}', color=color1, linestyle='-')
    plt.plot(recalls_vit, precisions_vit, label=f'TransUnet AP={ap_vit:.4f}', color=color2, linestyle='-')
    plt.plot(recalls_unet, precisions_unet, label=f'U-Net AP={ap_unet:.4f}', color=color3, linestyle='-')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'1PR_comparison_{fname}.png')
    plt.show()

def plot_roc_curve(gx, fp_vit, fp_ds,fp_unet, fname, avg_mcc):
    gx_binary = np.clip(np.rint(gx), 0, 1)
    fpr_vit, tpr_vit, thresholds_vit = roc_curve(gx_binary, fp_vit, pos_label=1)
    fpr_ds, tpr_ds, thresholds_ds = roc_curve(gx_binary, fp_ds, pos_label=1)
    fpr_unet, tpr_unet, thresholds_unet = roc_curve(gx_binary, fp_unet, pos_label=1)    
    roc_auc_vit = auc(fpr_vit, tpr_vit)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    roc_auc_unet = auc(fpr_unet, tpr_unet)

    plt.figure()
    plt.plot(fpr_ds, tpr_ds, label=f'DS-TransUnet AUC={roc_auc_ds:.4f}',color=color1, linestyle='-')
    plt.plot(fpr_vit, tpr_vit, label=f'TransUnet AUC={roc_auc_vit:.4f}',color=color2, linestyle='-')
    plt.plot(fpr_unet, tpr_unet, label=f'U-Net AUC={roc_auc_vit:.4f}',color=color3, linestyle='-')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'1ROC_comparison_{fname}.png')
    plt.show()

if __name__ == "__main__":
    main(["checkpointTUnet.43.pth","checkpointDS.43.pth","checkpointUNet.43.pth"])  # 调用函数并传入模型路径
