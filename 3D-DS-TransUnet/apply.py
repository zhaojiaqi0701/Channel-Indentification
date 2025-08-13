# import os
# import subprocess
# import sys
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from datUtils import *
# from unet3 import UNet3D  # 假设UNet3D是你的模型类定义

# def main(argv):
#     loadModel(argv[0])
#     goFakeValidation()
#     # goJie()

# def loadModel(mk):
#     global model
#     model = UNet3D()  # Initialize your U-Net model
#     checkpoint = torch.load('/root/autodl-tmp/Channel-checkUNet1/checkpointUNet.43.pth')
#     model.load_state_dict(checkpoint['state_dict'])  # 使用 'state_dict' 键
#     model.eval()
    
# def run_goDisplay(args):
#     with open('goDisplay1', 'r') as f:
#         command = f.read().strip()

#     # 替换 $* 为实际参数
#     command = command.replace('$*', ' '.join(args))

#     # 分割命令和参数
#     command_parts = command.split()

#     # 使用 subprocess 运行命令
#     subprocess.run(command_parts, check=True)


# def calculate_metrics(pred, label):
#     """Calculate accuracy, recall, IOU, and Dice coefficient"""
#     pred = pred.detach().cpu().numpy()
#     label = label.detach().cpu().numpy()

#     tp = np.sum((pred == 1) & (label == 1))
#     tn = np.sum((pred == 0) & (label == 0))
#     fp = np.sum((pred == 1) & (label == 0))
#     fn = np.sum((pred == 0) & (label == 1))

#     accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
#     recall = tp / (tp + fn + 1e-10)
#     iou = tp / (tp + fp + fn + 1e-10)
#     dice = 2 * tp / (2 * tp + fp + fn + 1e-10)

#     return accuracy, recall, iou, dice

# def goFakeValidation():
#     n1, n2, n3 = 256, 256, 256
#     seisPath = "/root/autodl-tmp/data/validationUNet/nx/"
#     lxpath = "/root/autodl-tmp/data/validationUNet/lx/"
#     predPath = "/root/autodl-tmp/data/validationUNet/px/"
#     ks = [201, 202, 203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219]


#     for k in ks:
#         fname = str(k)
#         gx = loadData(n1, n2, n3, seisPath, fname + '.dat')
#         gs = np.reshape(gx, (1, 1, n1, n2, n3))  # 修改为PyTorch格式
#         gs = torch.from_numpy(gs).float()

#         # label
#         lx = loadData1(n1, n2, n3, lxpath, fname + '.dat')
#         ls = np.reshape(lx, (1, 1, n1, n2, n3))
#         ls = torch.from_numpy(ls).float()

#         with torch.no_grad():
#             fp = model(gs)
#         fp = fp[0, 0, :, :, :].cpu().numpy()
#         ft = np.transpose(fp)
#         ft.tofile(predPath + fname + '.dat', format="%4")
#         np.save(predPath + fname + "_predictions.npy", ft)

#         # 计算精度、召回率、IOU 和 Dice 系数
#         binary_fp = (fp > 0.5).astype(int)
#         binary_ls = (ls[0, 0, :, :, :].cpu().numpy() > 0.5).astype(int)
#         accuracy, recall, iou, dice = calculate_metrics(torch.from_numpy(binary_fp), torch.from_numpy(binary_ls))

#         print(
#             f"Metrics for chunk {k}: Accuracy={accuracy * 100:.2f}%, Recall={recall * 100:.2f}%, IOU={iou * 100:.2f}%, Dice={dice * 100:.2f}%")

#         # 保存每个切片的指标到文件
#         with open(f"{fname}_metricsUnet.txt", 'a') as f:
#             f.write(f"Chunk {k} metrics:\n")
#             f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
#             f.write(f"Recall: {recall * 100:.2f}%\n")
#             f.write(f"IOU: {iou * 100:.2f}%\n")
#             f.write(f"Dice: {dice * 100:.2f}%\n")

#         # run_goDisplay(['valid', fname])  # Display or further processing





# import os
# import subprocess
# import sys
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import time
# from datUtils import *
# from unet3 import UNet3D  # 假设UNet3D是你的模型类定义

# def main(argv):
#     loadModel(argv[0])
#     goFakeValidation()

# def loadModel(mk):
#     global model
#     model = UNet3D()  # Initialize your U-Net model
#     checkpoint = torch.load('/root/autodl-tmp/Channel-checkUNet1/checkpointUNet.43.pth')
#     model.load_state_dict(checkpoint['state_dict'])  # 使用 'state_dict' 键
#     model.eval()
    
# def calculate_metrics(pred, label):
#     """Calculate accuracy, recall, IOU, Dice coefficient, and F1 score"""
#     pred = pred.detach().cpu().numpy()
#     label = label.detach().cpu().numpy()

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

#     return accuracy, recall, iou, dice, f1_score

# def goFakeValidation():
#     n1, n2, n3 = 256, 256, 256
#     seisPath = "/root/autodl-tmp/data/validationUNet/nx/"
#     lxpath = "/root/autodl-tmp/data/validationUNet/lx/"
#     predPath = "/root/autodl-tmp/data/validationUNet/px/"
#     ks = [251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]

#     for k in ks:
#         start_time = time.time()  # 记录开始时间

#         fname = str(k)
#         gx = loadData(n1, n2, n3, seisPath, fname + '.dat')
#         gs = np.reshape(gx, (1, 1, n1, n2, n3))  # 修改为PyTorch格式
#         gs = torch.from_numpy(gs).float()

#         # label
#         lx = loadData1(n1, n2, n3, lxpath, fname + '.dat')
#         ls = np.reshape(lx, (1, 1, n1, n2, n3))
#         ls = torch.from_numpy(ls).float()

#         with torch.no_grad():
#             fp = model(gs)
#         fp = fp[0, 0, :, :, :].cpu().numpy()
#         ft = np.transpose(fp)
#         ft.tofile(predPath + fname + '.dat', format="%4")
#         np.save(predPath + fname + "_predictions.npy", ft)

#         # 计算精度、召回率、IOU、Dice系数和F1分数
#         binary_fp = (fp > 0.5).astype(int)
#         binary_ls = (ls[0, 0, :, :, :].cpu().numpy() > 0.5).astype(int)
#         accuracy, recall, iou, dice, f1_score = calculate_metrics(torch.from_numpy(binary_fp), torch.from_numpy(binary_ls))

#         end_time = time.time()  # 记录结束时间
#         elapsed_time = end_time - start_time  # 计算耗时

#         print(
#             f"Metrics for chunk {k}: Accuracy={accuracy * 100:.2f}%, Recall={recall * 100:.2f}%, "
#             f"IOU={iou * 100:.2f}%, Dice={dice * 100:.2f}%, F1 Score={f1_score * 100:.2f}%, Time={elapsed_time:.2f}s")

#         # 保存每个切片的指标和耗时到文件
#         with open(f"{fname}_metricsUnet.txt", 'a') as f:
#             f.write(f"Chunk {k} metrics:\n")
#             f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
#             f.write(f"Recall: {recall * 100:.2f}%\n")
#             f.write(f"IOU: {iou * 100:.2f}%\n")
#             f.write(f"Dice: {dice * 100:.2f}%\n")
#             f.write(f"F1 Score: {f1_score * 100:.2f}%\n")
#             f.write(f"Time: {elapsed_time:.2f}s\n")


# # import subprocess
# # import torch
# # from datUtils import *
# # from unet.unet3 import UNet3D  # 确保你有这个模块


# # def main(argv):
# #     loadModel(argv[0])
# #     # goFakeValidation()
# #     goJie()
# # def loadModel(mk):
# #     global model
# #     model = UNet3D()  # 初始化你的模型
# #     checkpoint = torch.load(r"D:\anaconda3\envs\KarstSeg3D-master\unet\check\checkpoint7.50 (1).pth")
# #     model.load_state_dict(checkpoint['state_dict'])
# #     model.eval()

# # def run_goDisplay(args):
# #     with open('goDisplay1', 'r') as f:
# #         command = f.read().strip()

# #     # 替换 $* 为实际参数
# #     command = command.replace('$*', ' '.join(args))

# #     # 分割命令和参数
# #     command_parts = command.split()

# #     # 使用 subprocess 运行命令
# #     subprocess.run(command_parts, check=True)

# # def calculate_metrics(pred, label):
# #     """计算精确度、召回率、IOU 和 Dice 系数"""
# #     tp = torch.sum((pred == 1) & (label == 1)).item()
# #     tn = torch.sum((pred == 0) & (label == 0)).item()
# #     fp = torch.sum((pred == 1) & (label == 0)).item()
# #     fn = torch.sum((pred == 0) & (label == 1)).item()

# #     accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
# #     recall = tp / (tp + fn + 1e-10)
# #     iou = tp / (tp + fp + fn + 1e-10)
# #     dice = 2 * tp / (2 * tp + fp + fn + 1e-10)

# #     return accuracy, recall, iou, dice

# # def goFakeValidation():
# #     n1, n2, n3 = 256, 256, 256
# #     seisPath = "../data/validation/nx/"
# #     lxpath = "../data/validation/lx/"
# #     predPath = "../data/validation/px-self-UNet/"
# #     ks = [101, 102, 103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119]

# #     for k in ks:
# #         fname = str(k)
# #         gx = loadData(n1, n2, n3, seisPath, fname + '.dat')
# #         gs = np.reshape(gx, (1, 1, n1, n2, n3))  # 修改为PyTorch格式
# #         gs = torch.from_numpy(gs).float()

# #         # label
# #         lx = loadData(n1, n2, n3, lxpath, fname + '.dat')
# #         ls = np.reshape(lx, (1, 1, n1, n2, n3))
# #         ls = torch.from_numpy(ls).float()

# #         with torch.no_grad():
# #             fp = model(gs)
# #         fp = fp[0, 0, :, :, :].cpu().numpy()
# #         ft = np.transpose(fp)
# #         ft.tofile(predPath + fname + '.dat', format="%4")
# #         np.save(predPath + fname + "_predictions.npy", ft)

# #         # 计算精度、召回率、IOU 和 Dice 系数
# #         binary_fp = (fp > 0.5).astype(int)
# #         binary_ls = (ls[0, 0, :, :, :].cpu().numpy() > 0.5).astype(int)
# #         accuracy, recall, iou, dice = calculate_metrics(torch.from_numpy(binary_fp), torch.from_numpy(binary_ls))

# #         print(
# #             f"Metrics for chunk {k}: Accuracy={accuracy * 100:.2f}%, Recall={recall * 100:.2f}%, IOU={iou * 100:.2f}%, Dice={dice * 100:.2f}%")

# #         # 保存每个切片的指标到文件
# #         with open(f"{fname}_metricsUnet.txt", 'a') as f:
# #             f.write(f"Chunk {k} metrics:\n")
# #             f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
# #             f.write(f"Recall: {recall * 100:.2f}%\n")
# #             f.write(f"IOU: {iou * 100:.2f}%\n")
# #             f.write(f"Dice: {dice * 100:.2f}%\n")

# #         # run_goDisplay(['valid', fname])


# # def goJie():
# #     fname = "MIG_ALLAGC-1"
# #     n1, n2, n3 =128,256,512
# #     fpath = r"D:\anaconda3\envs\KarstSeg3D-master\data/"

# #     # 加载并重新调整地震数据的形状
# #     gx = np.load(r"D:\anaconda3\envs\KarstSeg3D-master\data\MIG_ALLAGC-1.npy")
# #     gx = np.reshape(gx, (1, 1, n1, n2, n3))
# #     gx = torch.from_numpy(gx).float()

# #     # 初始化fx为float64
# #     fx = np.zeros((n1, n2, n3), dtype=np.float64)

# #     # 使用模型进行预测
# #     with torch.no_grad():
# #         fp = model(gx)

# #     # 将预测结果从张量转换为NumPy数组，并存储到fx中
# #     fx[:, :, :] = fp[0, 0, :, :, :].cpu().numpy().astype(np.float64)
# #     fp = np.transpose(fx)
# #     # 二值化处理并计算准确率
# #     binary_fp = (fx > 0.5).astype(int)
# #     binary_gx = (gx[0, 0, :, :, :].cpu().numpy() > 0.5).astype(int)
# #     accuracy = np.mean(binary_fp == binary_gx)
# #     print(f"准确率: {accuracy * 100:.2f}%")

# #     # 保存结果到文件
# #     fp.tofile(fpath + "50-MIG_ALLAGC-1-UNet.dat")
# #     np.save(fpath + "50-MIG_ALLAGC-1-UNet.npy", fp)

# #     # # 运行显示命令
# #     run_goDisplay(['jie'])
# # #
# # # def goJie():
# # #     fname = "MIG_ALLAGC-1.npy"
# # #     n1, n2, n3 =128,256,512
# # #     fpath = r"D:\anaconda3\envs\KarstSeg3D-master\data/"
# # #
# # #     # 加载并重新调整地震数据的形状
# # #     gx = np.load(r"D:\anaconda3\envs\KarstSeg3D-master\data\MIG_ALLAGC-1.npy")
# # #     gx = np.reshape(gx, (1, 1, n1, n2, n3))
# # #     gx = torch.from_numpy(gx).float()
# # #
# # #     # 初始化fx为float64
# # #     fx = np.zeros((n1, n2, n3), dtype=np.float64)
# # #
# # #     # 使用模型进行预测
# # #     with torch.no_grad():
# # #         fp = model(gx)
# # #
# # #     # 将预测结果从张量转换为NumPy数组，并存储到fx中
# # #     fx[:, :, :] = fp[0, 0, :, :, :].cpu().numpy().astype(np.float64)
# # #
# # #     # 二值化处理并计算准确率
# # #     binary_fp = (fx > 0.5).astype(int)
# # #     binary_gx = (gx[0, 0, :, :, :].cpu().numpy() > 0.5).astype(int)
# # #     accuracy = np.mean(binary_fp == binary_gx)
# # #     print(f"准确率: {accuracy * 100:.2f}%")
# # #
# # #     # 保存结果到文件
# # #     fx.tofile(fpath + "50MIG_ALLAGC-1.dat")
# # #     np.save(fpath + "50MIG_ALLAGC-1.npy", fx)
# # #
# # #     # 运行显示命令
# # #     run_goDisplay(['jie'])

# def loadData(n1, n2, n3, path, fname):
#     gx = np.fromfile(path + fname, dtype=np.float32)
#     gm, gs = np.mean(gx), np.std(gx)
#     gx = (gx - gm) / gs
#     gx = np.reshape(gx, (n3, n2, n1))
#     gx = np.transpose(gx)
#     return gx

# # def loadData1(n1, n2, n3, path, fname):
# #     gx = np.fromfile(path + fname, dtype=np.int8)
# #     gm, gs = np.mean(gx), np.std(gx)
# #     gx = (gx - gm) / gs
# #     gx = np.reshape(gx, (n3, n2, n1))
# #     gx = np.transpose(gx)
# #     return gx
# def loadData1(n1, n2, n3, path, fname):
#     lx = np.fromfile(path + fname, dtype=np.int8)
#     lm, ls = np.mean(lx), np.std(lx)
#     lx = lx - lm
#     lx = lx / ls
#     lx = np.reshape(lx, (n3, n2, n1))
#     lx = np.transpose(lx)
#     return lx
# def sigmoid(x):
#     s = 1.0 / (1.0 + np.exp(-x))
#     return s

# if __name__ == '__main__':
#     main(sys.argv)
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from datUtils import *
from unet3 import UNet  # 假设UNet3D是你的模型类定义

def main(argv):
    loadModel(argv[0])
    goFakeValidation()
    # goJie()
def loadModel(mk):
    global model
    model = UNet()  # Initialize your U-Net model   UNet3D()变为UNet()因为将unet3.py里的函数名称进行更改
    checkpoint = torch.load('/root/autodl-tmp/Channel-checkUNet1/checkpointUNet.43.pth')
    model.load_state_dict(checkpoint['state_dict'])  # 使用 'state_dict' 键
    model.eval()
    
# def calculate_metrics(pred, label):
#     """Calculate accuracy, recall, IOU, Dice coefficient, precision, and F1 score"""
#     pred = pred.detach().cpu().numpy()
#     label = label.detach().cpu().numpy()

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

#     return accuracy, recall, iou, dice, precision, f1_score

# def goFakeValidation():
#     n1, n2, n3 = 256, 256, 256
#     seisPath = "/root/autodl-tmp/data/validationUNet/nx/"
#     lxpath = "/root/autodl-tmp/data/validationUNet/lx/"
#     predPath = "/root/autodl-tmp/data/validationUNet/px/"
#     ks = [251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]

#     for k in ks:
#         start_time = time.time()  # 记录开始时间

#         fname = str(k)
#         gx = loadData(n1, n2, n3, seisPath, fname + '.dat')
#         gs = np.reshape(gx, (1, 1, n1, n2, n3))  # 修改为PyTorch格式
#         gs = torch.from_numpy(gs).float()

#         # label
#         lx = loadData1(n1, n2, n3, lxpath, fname + '.dat')
#         ls = np.reshape(lx, (1, 1, n1, n2, n3))
#         ls = torch.from_numpy(ls).float()

#         with torch.no_grad():
#             fp = model(gs)
#         fp = fp[0, 0, :, :, :].cpu().numpy()
#         ft = np.transpose(fp)
#         ft.tofile(predPath + fname + '.dat', format="%4")
#         np.save(predPath + fname + "_predictions.npy", ft)

#         # 计算精度、召回率、IOU、Dice系数、Precision和F1分数
#         binary_fp = (fp > 0.5).astype(int)
#         binary_ls = (ls[0, 0, :, :, :].cpu().numpy() > 0.5).astype(int)
#         accuracy, recall, iou, dice, precision, f1_score = calculate_metrics(torch.from_numpy(binary_fp), torch.from_numpy(binary_ls))

#         end_time = time.time()  # 记录结束时间
#         elapsed_time = end_time - start_time  # 计算耗时

#         print(
#             f"Metrics for chunk {k}: Accuracy={accuracy * 100:.2f}%, Recall={recall * 100:.2f}%, "
#             f"IOU={iou * 100:.2f}%, Dice={dice * 100:.2f}%, Precision={precision * 100:.2f}%, "
#             f"F1 Score={f1_score * 100:.2f}%, Time={elapsed_time:.2f}s")

#         # 保存每个切片的指标和耗时到文件
#         with open(f"{fname}_metricsUnet.txt", 'a') as f:
#             f.write(f"Chunk {k} metrics:\n")
#             f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
#             f.write(f"Recall: {recall * 100:.2f}%\n")
#             f.write(f"IOU: {iou * 100:.2f}%\n")
#             f.write(f"Dice: {dice * 100:.2f}%\n")
#             f.write(f"Precision: {precision * 100:.2f}%\n")
#             f.write(f"F1 Score: {f1_score * 100:.2f}%\n")
#             f.write(f"Time: {elapsed_time:.2f}s\n")

def calculate_metrics(pred, label):
    """计算精确度、召回率、IOU 和 Dice 系数"""
    tp = torch.sum((pred == 1) & (label == 1)).item()
    tn = torch.sum((pred == 0) & (label == 0)).item()
    fp = torch.sum((pred == 1) & (label == 0)).item()
    fn = torch.sum((pred == 0) & (label == 1)).item()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-10)

    return accuracy, recall, iou, dice

def goFakeValidation():
    n1, n2, n3 = 256, 256, 256
    seisPath = "/root/autodl-tmp/data/validationUNet/nx/"
    lxpath = "/root/autodl-tmp/data/validationUNet/lx/"
    predPath = "/root/autodl-tmp/data/validationUNet/px/"
    ks = [251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]

    for k in ks:
        fname = str(k)
        gx = loadData(n1, n2, n3, seisPath, fname + '.dat')
        gs = np.reshape(gx, (1, 1, n1, n2, n3))  # 修改为PyTorch格式
        gs = torch.from_numpy(gs).float()

        # label
        lx = loadData1(n1, n2, n3, lxpath, fname + '.dat')
        ls = np.reshape(lx, (1, 1, n1, n2, n3))
        ls = torch.from_numpy(ls).float()

        with torch.no_grad():
            fp = model(gs)
        fp = fp[0, 0, :, :, :].cpu().numpy()
        ft = np.transpose(fp)
        ft.tofile(predPath + fname + '.dat', format="%4")
        np.save(predPath + fname + "_predictions.npy", ft)

        # 计算精度、召回率、IOU 和 Dice 系数
        binary_fp = (fp > 0.5).astype(int)
        binary_ls = (ls[0, 0, :, :, :].cpu().numpy() > 0.5).astype(int)
        accuracy, recall, iou, dice = calculate_metrics(torch.from_numpy(binary_fp), torch.from_numpy(binary_ls))

        print(
            f"Metrics for chunk {k}: Accuracy={accuracy * 100:.2f}%, Recall={recall * 100:.2f}%, IOU={iou * 100:.2f}%, Dice={dice * 100:.2f}%")

        # 保存每个切片的指标到文件
        with open(f"{fname}_metricsUnet.txt", 'a') as f:
            f.write(f"Chunk {k} metrics:\n")
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            f.write(f"Recall: {recall * 100:.2f}%\n")
            f.write(f"IOU: {iou * 100:.2f}%\n")
            f.write(f"Dice: {dice * 100:.2f}%\n")

import time

def calculate_accuracy(pred, label):
    """计算二值化预测的准确率"""
    binary_pred = (pred > 0.5).astype(int)  # 将预测结果二值化
    binary_label = (label > 0.5).astype(int)  # 将实际值二值化
    accuracy = np.mean(binary_pred == binary_label)  # 计算准确率
    return accuracy

# def goJie():
#     fname = "seismic_192-slice4.npy"
#     n1, n2, n3 = 128, 256, 256
#     input_size = 128
#     fpath = r"/root/autodl-tmp/data/"
    
#     # 开始计时
#     start_time = time.time()

#     # 加载地震数据
#     gx = np.load(fpath + fname)
#     gx = np.reshape(gx, (1, 1, n1, n2, n3))  # 转换为 PyTorch 格式
#     gx_torch = torch.from_numpy(gx).float()  # 转换为 PyTorch 张量

#     # 执行预测
#     with torch.no_grad():
#         fp = model(gx_torch)  # Fault prediction
#     fp = fp.cpu().numpy()  # 转换为 NumPy 数组

#     # 计算准确率
#     accuracy = calculate_accuracy(fp[0, 0, :, :, :], gx[0, 0, :, :, :])
#     print(f"Accuracy: {accuracy * 100:.2f}%")  # 打印准确率

#     # 保存预测结果
#     fx = np.zeros((n1, n2, n3), dtype=np.single)
#     fx[:, :, :] = fp[0, 0, :, :, :]
#     fx = np.transpose(fx)  # 转置以匹配保存格式
#     fx.tofile(fpath + "seismic-UNet.dat", format="%4")
#     output_file = fpath + "seismic-UNet.npy"
#     np.save(output_file, fx)
#     print(f"Prediction saved to {output_file}")

#     # 结束计时并打印耗时
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Time taken for prediction: {elapsed_time:.2f} seconds")

def goJie():
    # fname = "ChannelSeismic(128,512,512).npy"
    fname = "seismic_192-slice4.npy"
    # n1, n2, n3 = 128, 512, 512
    n1, n2, n3 = 128, 256, 256
    input_size = 128
    overlap = 4
    # fname = "Seismic_80Hz_new(128,384,128).npy"
    # # n1, n2, n3 = 128, 512, 512
    # n1, n2, n3 = 128,384,128
    # input_size = 128
    # overlap = 4
    fpath = r"/root/autodl-tmp/data/"
    # 开始计时
    start_time = time.time()

    # 加载地震数据
    gx = np.load(fpath + fname)
    gx = np.reshape(gx, (1, 1, n1, n2, n3))  # 转换为 PyTorch 格式
    gx_torch = torch.from_numpy(gx).float()  # 转换为 PyTorch 张量

    # 执行预测
    with torch.no_grad():
        fp = model(gx_torch)  # Fault prediction     
    fp = fp.cpu().numpy()  # 转换为 NumPy 数组

    # 计算准确率
    accuracy = calculate_accuracy(fp[0, 0, :, :, :], gx[0, 0, :, :, :])
    print(f"Accuracy: {accuracy * 100:.2f}%")  # 打印准确率

    # 保存预测结果
    fx = np.zeros((n1, n2, n3), dtype=np.single)
    fx[:, :, :] = fp[0, 0, :, :, :]
    fx = np.transpose(fx)  # 转置以匹配保存格式
    # fx.tofile(fpath + "ChannelSeismic-UNet43.dat", format="%4")
    # output_file = fpath + "ChannelSeismic-UNet43.npy"
    output_file = fpath + "ChannelSeismic-UNet43-probabilities.npy"

    np.save(output_file, fx)
    print(f"Prediction saved to {output_file}")

    # 结束计时并打印耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for prediction: {elapsed_time:.2f} seconds")
    
def loadData(n1, n2, n3, path, fname):
    gx = np.fromfile(path + fname, dtype=np.single)
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
def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s
if __name__ == '__main__':
    main(sys.argv)
