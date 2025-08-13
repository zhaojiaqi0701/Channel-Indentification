import os
import subprocess
import sys
import time  # 导入时间模块

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datUtils import *
from deeplabv3plus import DeepLabV3Plus
import configs as configs

# def main(argv):
#     loadModel(argv[0])
#     goFakeValidation()

# def loadModel(mk):
#     global model
#     model = DeepLabV3Plus(
#     in_channels=1, 
#     n_classes=1,
#     n_blocks=[3, 4, 23, 3],
#     atrous_rates=[6, 12, 18],
#     multi_grids=[1, 2, 4],
#     output_stride=16,
# )
#     checkpoint = torch.load("/root/autodl-tmp/Channel-checkDLab3plus/checkpointDLab3plus.43.pth")
#     model.load_state_dict(checkpoint['state_dict'])
#     model.eval()

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

# def calculate_metrics(pred, label):
#     """计算精确度、召回率、IOU、Dice 系数和 F1 分数"""
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
#     input_size = 128
#     overlap = 4
#     seisPath = "/root/autodl-tmp/data/validationDLab/nx/"
#     lxpath = "/root/autodl-tmp/data/validationDLab/lx/"
#     predPath = "/root/autodl-tmp/data/validationDLab/px/"
#     ks = [251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]

#     trilinear_weights = create_trilinear_weights(input_size, overlap)

#     for k in ks:
#         start_time = time.time()  # 记录开始时间

#         fname = str(k)
#         gx = loadData(n1, n2, n3, seisPath, fname + '.dat')
#         gx = np.reshape(gx, (1, 1, n1, n2, n3))
#         gx = torch.from_numpy(gx).float()
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
#         avg_accuracy, avg_recall, avg_iou, avg_dice, avg_f1_score = 0.0, 0.0, 0.0, 0.0, 0.0
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
#                         accuracy, recall, iou, dice, f1_score = calculate_metrics(binary_fp, binary_ls)
#                         chunk_metrics.append((accuracy, recall, iou, dice, f1_score))

#                         avg_accuracy += accuracy
#                         avg_recall += recall
#                         avg_iou += iou
#                         avg_dice += dice
#                         avg_f1_score += f1_score
#                         num_chunks += 1
#                         print(f"Metrics for chunk {num_chunks}: Accuracy={accuracy * 100:.2f}%, Recall={recall * 100:.2f}%, "
#                               f"IOU={iou * 100:.2f}%, Dice={dice * 100:.2f}%, F1 Score={f1_score * 100:.2f}%")

#                         fx[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += f[0, 0] * trilinear_weights
#                         weight[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += trilinear_weights

#             fx = np.divide(fx, weight, out=np.zeros_like(fx), where=weight != 0)
#             fx = np.transpose(fx)

#         fx.tofile(predPath + fname + '43.dat', format="%4")
#         np.save(predPath + fname + "43_predictions.npy", fx)
#         avg_accuracy /= num_chunks
#         avg_recall /= num_chunks
#         avg_iou /= num_chunks
#         avg_dice /= num_chunks
#         avg_f1_score /= num_chunks

#         end_time = time.time()
#         elapsed_time = end_time - start_time  # 计算每个数据体的处理时间

#         print(f"Average metrics for {fname}: Accuracy={avg_accuracy * 100:.2f}%, Recall={avg_recall * 100:.2f}%, "
#               f"IOU={avg_iou * 100:.2f}%, Dice={avg_dice * 100:.2f}%, F1 Score={avg_f1_score * 100:.2f}%, "
#               f"Time={elapsed_time:.2f}s")

#         with open(f"{fname}_metricsDLab.txt", 'a') as f:
#             f.write(f"Chunk metrics:\n")
#             for i, (acc, rec, iou, dice, f1) in enumerate(chunk_metrics):
#                 f.write(f"Chunk {i}: Accuracy={acc * 100:.2f}%, Recall={rec * 100:.2f}%, IOU={iou * 100:.2f}%, "
#                         f"Dice={dice * 100:.2f}%, F1 Score={f1 * 100:.2f}%\n")
#             f.write(f"\nAverage metrics: Accuracy={avg_accuracy * 100:.2f}%, Recall={avg_recall * 100:.2f}%, "
#                     f"IOU={avg_iou * 100:.2f}%, Dice={avg_dice * 100:.2f}%, F1 Score={avg_f1_score * 100:.2f}%\n")
#             f.write(f"Time for {fname}: {elapsed_time:.2f}s\n")


    
# # 三线性插值
# def goJie():
#     # fname = "seismic_192-slice4.npy"
#     fname = "MIG_ALLAGC-1.npy"
#     # n1, n2, n3 = 128, 256, 256
#     n1, n2, n3 = 128, 256, 512
#     input_size = 128
#     overlap = 2
#     fpath = r"/root/autodl-tmp/data/"
    
#     def calculate_step_size(n, input_size, overlap):  
#         step = input_size - overlap  
#         num_chunks = (n - input_size) // step + 1  
#         return step, num_chunks

#     def load_and_preprocess_data(filepath, shape):
#         gx = np.load(filepath)
#         gx = np.reshape(gx, (1, 1, *shape))  # 修改为PyTorch格式
#         gx = torch.from_numpy(gx).float()
#         return gx

#     def create_trilinear_weights(input_size, overlap):
#         weights = np.ones((input_size, input_size, input_size))
#         for i in range(input_size):
#             for j in range(input_size):
#                 for k in range(input_size):
#                     factor_i = 1.0
#                     factor_j = 1.0
#                     factor_k = 1.0
#                     if i < overlap:
#                         factor_i = (i + 1) / (overlap + 1)
#                     elif i >= input_size - overlap:
#                         factor_i = (input_size - i) / (overlap + 1)
#                     if j < overlap:
#                         factor_j = (j + 1) / (overlap + 1)
#                     elif j >= input_size - overlap:
#                         factor_j = (input_size - j) / (overlap + 1)
#                     if k < overlap:
#                         factor_k = (k + 1) / (overlap + 1)
#                     elif k >= input_size - overlap:
#                         factor_k = (input_size - k) / (overlap + 1)
#                     weights[i, j, k] = factor_i * factor_j * factor_k
#         return weights

#     def process_chunk(gx, start_n1, start_n2, start_n3, input_size, model):
#         end_n1 = start_n1 + input_size
#         end_n2 = start_n2 + input_size
#         end_n3 = start_n3 + input_size

#         g = gx[:, :, start_n1:end_n1, start_n2:end_n2, start_n3:end_n3]
#         if g.size(3) == input_size and g.size(4) == input_size:
#             f = model(g).cpu().numpy()[0, 0, :, :, :]
#             return f, (start_n1, end_n1), (start_n2, end_n2), (start_n3, end_n3)
#         else:
#             print("Warning: Invalid slice dimensions for model input.")
#             return None, None, None, None

#     step1, num_chunks_n1 = calculate_step_size(n1, input_size, overlap)
#     step2, num_chunks_n2 = calculate_step_size(n2, input_size, overlap)
#     step3, num_chunks_n3 = calculate_step_size(n3, input_size, overlap)
    
#     print('Steps:', step1, step2, step3)
#     print('Number of chunks:', num_chunks_n1, num_chunks_n2, num_chunks_n3)
    
#     trilinear_weights = create_trilinear_weights(input_size, overlap)
#     gx = load_and_preprocess_data(fpath + fname, (n1, n2, n3))
#     print(gx.shape)
    
#     fx = np.zeros((n1, n2, n3), dtype=np.float64)
#     weight = np.zeros_like(fx)
    
#     config = configs.get_r50_b16_config()
#     model = DA_Transformer(config, img_size=128, num_classes=21843, zero_head=False, vis=False)
#     checkpoint = torch.load("/root/autodl-tmp/check2/checkpoint6.50.pth")
#     model.load_state_dict(checkpoint['state_dict'])
#     model.eval()
#     avg_accuracy = 0.0
#     num_chunks = 0
#     chunk_accuracies = []   
#     with torch.no_grad():
#         for i in range(num_chunks_n1):
#             for j in range(num_chunks_n2):
#                 for k in range(num_chunks_n3):
#                     start_n1 = i * step1
#                     start_n2 = j * step2
#                     start_n3 = k * step3

#                     f, range_n1, range_n2, range_n3 = process_chunk(gx, start_n1, start_n2, start_n3, input_size, model)
#                     if f is not None:
#                         start_n1, end_n1 = range_n1
#                         start_n2, end_n2 = range_n2
#                         start_n3, end_n3 = range_n3
                        
#                         weights_slice = trilinear_weights[:end_n1-start_n1, :end_n2-start_n2, :end_n3-start_n3]
#                         fx[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += f * weights_slice
#                         weight[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += weights_slice
#                         # 计算当前切片的精确度
#                         binary_fp = (f > 0.5).astype(int)
#                         binary_ls = (gx[0, 0, start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] > 0.5).cpu().numpy().astype(int)
#                         accuracy = np.mean(binary_fp == binary_ls)
#                         chunk_accuracies.append(accuracy)
#                         avg_accuracy += accuracy
#                         num_chunks += 1
#                         print(f"Accuracy for chunk {num_chunks}: {accuracy * 100:.2f}%")
#     # 对所有区域进行平滑处理
#     fx = np.divide(fx, weight, out=np.zeros_like(fx), where=weight != 0)
#     fx = np.transpose(fx)
#     avg_accuracy /= num_chunks
#     print(f"Average accuracy: {avg_accuracy * 100:.2f}%")    
#     fx.tofile(fpath + "50your_output_file-DA.dat", format="%4")
#     output_file = fpath + "50your_output_file-DA-6.50.npy"
#     np.save(output_file, fx)
#     run_goDisplay(['jie'])
# def goHongliu():
#     fname = "gg.dat"
#     n1, n2, n3 = 256, 256, 256
#     fpath = "/media/xinwu/disk-2/karstFW/hongliu/"
#     gx = loadData(n1, n2, n3, fpath, fname)  # load seismic
#     gx = np.reshape(gx, (1, 1, n1, n2, n3))  # 修改为PyTorch格式
#     gx = torch.from_numpy(gx).float()
#     with torch.no_grad():
#         fp = model(gx)  # fault prediction
#     fp = fp.cpu().numpy()
#     fx = np.zeros((n1, n2, n3), dtype=np.single)
#     fx[:, :, :] = fp[0, 0, :, :, :]
#     fx = np.transpose(fx)
#     fx.tofile(fpath + "fp.dat", format="%4")
#     run_goDisplay(['hongliu'])

# def loadData(n1, n2, n3, path, fname):
#     gx = np.fromfile(path + fname, dtype=np.single)
#     gm, gs = np.mean(gx), np.std(gx)
#     gx = gx - gm
#     gx = gx / gs
#     gx = np.reshape(gx, (n1, n2, n3))
#     gx = np.transpose(gx)
#     # gx_chunks = []
#     # for i in range(0, n3, 64):
#     #     for j in range(0, n2, 64):
#     #         for k in range(0, n1, 64):
#     #             chunk = gx[k:k+64, j:j+64, i:i+64]  # 从原始数据中切割出 64x64x64 的块
#     #             chunk = np.transpose(chunk)
#     #             gx_chunks.append(chunk)
#     # return gx_chunks
#     return gx
# # def loadData1(n1, n2, n3, path, fname):
# #     lx = np.fromfile(path + fname, dtype=np.single)
# #     lm, ls = np.mean(lx), np.std(lx)
# #     lx = lx - lm
# #     lx = lx / ls
# #     lx = np.reshape(lx, (n3, n2, n1))
# #     # lx = np.transpose(lx)
# #     lx_chunks = []
# #     for i in range(0, n3, 64):
# #         for j in range(0, n2, 64):
# #             for k in range(0, n1, 64):
# #                 chunk = lx[k:k+64, j:j+64, i:i+64]  # 从原始数据中切割出 64x64x64 的块
# #                 chunk = np.transpose(chunk)
# #                 lx_chunks.append(chunk)
# #     return lx_chunks
# def loadData1(n1, n2, n3, path, fname):
#     lx = np.fromfile(path + fname, dtype=np.int8)
#     lm, ls = np.mean(lx), np.std(lx)
#     lx = lx - lm
#     lx = lx / ls
#     lx = np.reshape(lx, (n3, n2, n1))
#     lx = np.transpose(lx)
#     return lx
# def loadDatax(n1, n2, n3, path, fname):
#     gx = np.fromfile(path + fname, dtype=np.single)
#     gx = np.reshape(gx, (n3, n2, n1))
#     gx = np.transpose(gx)
#     return gx

# def sigmoid(x):
#     s = 1.0 / (1.0 + np.exp(-x))
#     return s

# # def plot2d(gx, fp, fx, at=1, png=None):
# #     fig = plt.figure(figsize=(15, 5))
# #     ax = fig.add_subplot(131)
# #     ax.imshow(gx, vmin=-2, vmax=2, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
# #     # ax = fig.add_subplot(132)
# #     # ax.imshow(fp, vmin=0, vmax=1, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
# #     ax = fig.add_subplot(133)
# #     ax.imshow(fx, vmin=0, vmax=1.0, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
# #     if png:
# #         plt.savefig(getPngDir() + png + '.png')
# #     plt.tight_layout()
# #     plt.show()

# if __name__ == '__main__':
#     main(sys.argv)

def main(argv):
    loadModel(argv[0])
    goFakeValidation()
    
def loadModel(mk):
    global model
    model = DeepLabV3Plus(
    in_channels=1, 
    n_classes=1,
    n_blocks=[3, 4, 23, 3],
    atrous_rates=[6, 12, 18],
    multi_grids=[1, 2, 4],
    output_stride=16,
)
    checkpoint = torch.load("/root/autodl-tmp/Channel-checkDLab3plus/checkpointDLab3plus.43.pth")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

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
def calculate_metrics(pred, label):
    """Calculate accuracy, recall, IOU, Dice coefficient, precision, and F1 score"""
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()

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

    return accuracy, recall, iou, dice, precision, f1_score

def goFakeValidation():
    n1, n2, n3 = 256, 256, 256
    seisPath = "/root/autodl-tmp/data/validationUNet/nx/"
    lxpath = "/root/autodl-tmp/data/validationUNet/lx/"
    predPath = "/root/autodl-tmp/data/validationUNet/px/"
    ks = [251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]

    for k in ks:
        start_time = time.time()  # 记录开始时间

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

        # 计算精度、召回率、IOU、Dice系数、Precision和F1分数
        binary_fp = (fp > 0.5).astype(int)
        binary_ls = (ls[0, 0, :, :, :].cpu().numpy() > 0.5).astype(int)
        accuracy, recall, iou, dice, precision, f1_score = calculate_metrics(torch.from_numpy(binary_fp), torch.from_numpy(binary_ls))

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时

        print(
            f"Metrics for chunk {k}: Accuracy={accuracy * 100:.2f}%, Recall={recall * 100:.2f}%, "
            f"IOU={iou * 100:.2f}%, Dice={dice * 100:.2f}%, Precision={precision * 100:.2f}%, "
            f"F1 Score={f1_score * 100:.2f}%, Time={elapsed_time:.2f}s")

        # 保存每个切片的指标和耗时到文件
        with open(f"{fname}_metricsDLab.txt", 'a') as f:
            f.write(f"Chunk {k} metrics:\n")
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            f.write(f"Recall: {recall * 100:.2f}%\n")
            f.write(f"IOU: {iou * 100:.2f}%\n")
            f.write(f"Dice: {dice * 100:.2f}%\n")
            f.write(f"Precision: {precision * 100:.2f}%\n")
            f.write(f"F1 Score: {f1_score * 100:.2f}%\n")
            f.write(f"Time: {elapsed_time:.2f}s\n")
def loadData(n1, n2, n3, path, fname):
    gx = np.fromfile(path + fname, dtype=np.float32)
    gm, gs = np.mean(gx), np.std(gx)
    gx = (gx - gm) / gs
    gx = np.reshape(gx, (n3, n2, n1))
    gx = np.transpose(gx)
    return gx

# def loadData1(n1, n2, n3, path, fname):
#     gx = np.fromfile(path + fname, dtype=np.int8)
#     gm, gs = np.mean(gx), np.std(gx)
#     gx = (gx - gm) / gs
#     gx = np.reshape(gx, (n3, n2, n1))
#     gx = np.transpose(gx)
#     return gx
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
