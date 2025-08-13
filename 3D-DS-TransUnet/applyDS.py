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
from DS_TransUNet3D import UNet3D
import configs as configs

def main(argv):
    loadModel(argv[0])
    # goFakeValidation()
    goJie()
def loadModel(mk):
    global model
    model = UNet3D(128, 1)
    checkpoint = torch.load("/root/autodl-tmp/Channel-checkDS/checkpointDS.43.pth")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

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

    return accuracy, recall, iou, dice, precision, f1_score

# def goFakeValidation():
#     n1, n2, n3 = 256, 256, 256
#     input_size = 128
#     overlap = 4
#     seisPath = "/root/autodl-tmp/data/validationDS/nx/"
#     lxpath = "/root/autodl-tmp/data/validationDS/lx/"
#     predPath = "/root/autodl-tmp/data/validationDS/px/"
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
#         avg_accuracy, avg_recall, avg_iou, avg_dice, avg_precision, avg_f1_score = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
#                         accuracy, recall, iou, dice, precision, f1_score = calculate_metrics(binary_fp, binary_ls)
#                         chunk_metrics.append((accuracy, recall, iou, dice, precision, f1_score))

#                         avg_accuracy += accuracy
#                         avg_recall += recall
#                         avg_iou += iou
#                         avg_dice += dice
#                         avg_precision += precision
#                         avg_f1_score += f1_score
#                         num_chunks += 1
#                         print(f"Metrics for chunk {num_chunks}: Accuracy={accuracy * 100:.2f}%, Recall={recall * 100:.2f}%, "
#                               f"IOU={iou * 100:.2f}%, Dice={dice * 100:.2f}%, Precision={precision * 100:.2f}%, "
#                               f"F1 Score={f1_score * 100:.2f}%")

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
#         avg_precision /= num_chunks
#         avg_f1_score /= num_chunks

#         end_time = time.time()
#         elapsed_time = end_time - start_time  # 计算每个数据体的处理时间

#         print(f"Average metrics for {fname}: Accuracy={avg_accuracy * 100:.2f}%, Recall={avg_recall * 100:.2f}%, "
#               f"IOU={avg_iou * 100:.2f}%, Dice={avg_dice * 100:.2f}%, Precision={avg_precision * 100:.2f}%, "
#               f"F1 Score={avg_f1_score * 100:.2f}%, Time={elapsed_time:.2f}s")
        
#         with open(f"{fname}_metricsDS.txt", 'a') as f:
#             f.write(f"Chunk metrics:\n")
#             for i, (acc, rec, iou, dice, prec, f1) in enumerate(chunk_metrics):
#                 f.write(f"Chunk {i}: Accuracy={acc * 100:.2f}%, Recall={rec * 100:.2f}%, IOU={iou * 100:.2f}%, "
#                         f"Dice={dice * 100:.2f}%, Precision={avg_precision * 100:.2f}%, F1 Score={f1 * 100:.2f}%\n")
#             f.write(f"\nAverage metrics: Accuracy={avg_accuracy * 100:.2f}%, Recall={avg_recall * 100:.2f}%, "
#                     f"IOU={avg_iou * 100:.2f}%, Dice={avg_dice * 100:.2f}%,  Precision={avg_precision * 100:.2f}%, F1 Score={avg_f1_score * 100:.2f}%\n")
#             f.write(f"Time for {fname}: {elapsed_time:.2f}s\n")
def goFakeValidation():
    n1, n2, n3 = 256, 256, 256
    input_size = 128
    overlap = 4
    seisPath = "/root/autodl-tmp/data/validationDS/nx/"
    lxpath = "/root/autodl-tmp/data/validationDS/lx/"
    predPath = "/root/autodl-tmp/data/validationDS/px/"
    ks = [251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]

    trilinear_weights = create_trilinear_weights(input_size, overlap)

    for k in ks:
        start_time = time.time()  # 记录开始时间

        fname = str(k)
        gx = loadData(n1, n2, n3, seisPath, fname + '.dat')
        gx = np.reshape(gx, (1, 1, n1, n2, n3))
        gx = torch.from_numpy(gx).float()
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

        fx = np.zeros((n1, n2, n3), dtype=np.single)
        weight = np.zeros((n1, n2, n3), dtype=np.single)
        avg_accuracy, avg_recall, avg_iou, avg_dice, avg_precision, avg_f1_score = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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

                        f = model(g)

                        if isinstance(f, (tuple, list)):
                            f = f[0]

                        f = f.cpu().numpy()  # 将 f 转换为 numpy 数组
                        f1 = torch.from_numpy(f)  # 转换回 Tensor 类型

                        l = l.cpu().numpy()

                        # 如果是二分类任务，应用 sigmoid 将输出转换为概率
                        probabilities = torch.sigmoid(f1)  # 应用 sigmoid 转换为概率值
                        probabilities = probabilities.cpu().numpy()

                        binary_fp = (probabilities > 0.5).astype(int)  # 阈值设置为0.5
                        binary_ls = (l > 0.5).astype(int)
                        accuracy, recall, iou, dice, precision, f1_score = calculate_metrics(binary_fp, binary_ls)
                        chunk_metrics.append((accuracy, recall, iou, dice, precision, f1_score))

                        avg_accuracy += accuracy
                        avg_recall += recall
                        avg_iou += iou
                        avg_dice += dice
                        avg_precision += precision
                        avg_f1_score += f1_score
                        num_chunks += 1
                        print(f"Metrics for chunk {num_chunks}: Accuracy={accuracy * 100:.2f}%, Recall={recall * 100:.2f}%, "
                              f"IOU={iou * 100:.2f}%, Dice={dice * 100:.2f}%, Precision={precision * 100:.2f}%, "
                              f"F1 Score={f1_score * 100:.2f}%")

                        fx[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += probabilities[0, 0] * trilinear_weights
                        weight[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += trilinear_weights

            fx = np.divide(fx, weight, out=np.zeros_like(fx), where=weight != 0)
            fx = np.transpose(fx)
        fx.tofile(predPath + fname + '43.dat', format="%4")
        np.save(predPath + fname + "43_predictions.npy", fx)  # 保存原始预测
        np.save(predPath + fname + "43_probabilities.dat", fx)  # 保存概率值
        avg_accuracy /= num_chunks
        avg_recall /= num_chunks
        avg_iou /= num_chunks
        avg_dice /= num_chunks
        avg_precision /= num_chunks
        avg_f1_score /= num_chunks

        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算每个数据体的处理时间

        print(f"Average metrics for {fname}: Accuracy={avg_accuracy * 100:.2f}%, Recall={avg_recall * 100:.2f}%, "
              f"IOU={avg_iou * 100:.2f}%, Dice={avg_dice * 100:.2f}%, Precision={avg_precision * 100:.2f}%, "
              f"F1 Score={avg_f1_score * 100:.2f}%, Time={elapsed_time:.2f}s")
        
        with open(f"{fname}_metricsDS.txt", 'a') as f:
            f.write(f"Chunk metrics:\n")
            for i, (acc, rec, iou, dice, prec, f1) in enumerate(chunk_metrics):
                f.write(f"Chunk {i}: Accuracy={acc * 100:.2f}%, Recall={rec * 100:.2f}%, IOU={iou * 100:.2f}%, "
                        f"Dice={dice * 100:.2f}%, Precision={avg_precision * 100:.2f}%, F1 Score={f1 * 100:.2f}%\n")
            f.write(f"\nAverage metrics: Accuracy={avg_accuracy * 100:.2f}%, Recall={avg_recall * 100:.2f}%, "
                    f"IOU={avg_iou * 100:.2f}%, Dice={avg_dice * 100:.2f}%,  Precision={avg_precision * 100:.2f}%, F1 Score={avg_f1_score * 100:.2f}%\n")
            f.write(f"Time for {fname}: {elapsed_time:.2f}s\n")

import time

def goJie():
    # fname = "ChannelSeismic(128,512,512).npy"
    fname = "seismic_192-slice4.npy"
    # n1, n2, n3 = 128, 512, 512
    n1, n2, n3 = 128, 256, 256
    input_size = 128
    overlap = 4
    fpath = r"/root/autodl-tmp/data/"
    
    # 记录开始时间
    start_time = time.time()

    def calculate_step_size(n, input_size, overlap):  
        step = input_size - overlap  
        num_chunks = (n - input_size) // step + 1  
        return step, num_chunks

    def load_and_preprocess_data(filepath, shape):
        gx = np.load(filepath)
        gx = np.reshape(gx, (1, 1, *shape))  # Modify to PyTorch-compatible format
        gx = torch.from_numpy(gx).float()
        return gx

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

#     def process_chunk(gx, start_n1, start_n2, start_n3, input_size, model):
#         end_n1 = start_n1 + input_size
#         end_n2 = start_n2 + input_size
#         end_n3 = start_n3 + input_size

#         g = gx[:, :, start_n1:end_n1, start_n2:end_n2, start_n3:end_n3]
#         if g.size(3) == input_size and g.size(4) == input_size:
#             output = model(g)
#             if isinstance(output, tuple):  # Handle tuple output
#                 output = output[0]
#             f = output.cpu().numpy()[0, 0, :, :, :]
#             return f, (start_n1, end_n1), (start_n2, end_n2), (start_n3, end_n3)
#         else:
#             print("Warning: Invalid slice dimensions for model input.")
#             return None, None, None, None

# 保存概率值
    def process_chunk(gx, start_n1, start_n2, start_n3, input_size, model):
        end_n1 = start_n1 + input_size
        end_n2 = start_n2 + input_size
        end_n3 = start_n3 + input_size

        g = gx[:, :, start_n1:end_n1, start_n2:end_n2, start_n3:end_n3]
        if g.size(3) == input_size and g.size(4) == input_size:
            output = model(g)
            if isinstance(output, tuple):  # Handle tuple output
                output = output[0]

            # Apply sigmoid to get probabilities
            f = torch.sigmoid(output)  # Sigmoid to convert to probabilities

            # Convert tensor to numpy array and select the first channel/slice
            f = f.cpu().numpy()[0, 0, :, :, :]
            return f, (start_n1, end_n1), (start_n2, end_n2), (start_n3, end_n3)
        else:
            print("Warning: Invalid slice dimensions for model input.")
            return None, None, None, None
    step1, num_chunks_n1 = calculate_step_size(n1, input_size, overlap)
    step2, num_chunks_n2 = calculate_step_size(n2, input_size, overlap)
    step3, num_chunks_n3 = calculate_step_size(n3, input_size, overlap)
    
    print('Steps:', step1, step2, step3)
    print('Number of chunks:', num_chunks_n1, num_chunks_n2, num_chunks_n3)
    
    trilinear_weights = create_trilinear_weights(input_size, overlap)
    gx = load_and_preprocess_data(fpath + fname, (n1, n2, n3))
    print(f"Loaded input data with shape: {gx.shape}")
    
    fx = np.zeros((n1, n2, n3), dtype=np.float32)
    weight = np.zeros_like(fx)
    
    # Load model and checkpoint
    model = UNet3D(128, 1)
    checkpoint_path = "/root/autodl-tmp/Channel-checkDS/checkpointDS.43.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    
    avg_accuracy = 0.0
    num_chunks = 0
    chunk_accuracies = []
    
    with torch.no_grad():
        for i in range(num_chunks_n1):
            for j in range(num_chunks_n2):
                for k in range(num_chunks_n3):
                    start_n1 = i * step1
                    start_n2 = j * step2
                    start_n3 = k * step3

                    f, range_n1, range_n2, range_n3 = process_chunk(gx, start_n1, start_n2, start_n3, input_size, model)
                    if f is not None:
                        start_n1, end_n1 = range_n1
                        start_n2, end_n2 = range_n2
                        start_n3, end_n3 = range_n3
                        
                        weights_slice = trilinear_weights[:end_n1-start_n1, :end_n2-start_n2, :end_n3-start_n3]
                        fx[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += f * weights_slice
                        weight[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += weights_slice
                        
                        # Calculate accuracy for the current chunk
                        binary_fp = (f > 0.5).astype(int)
                        binary_ls = (gx[0, 0, start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] > 0.5).cpu().numpy().astype(int)
                        accuracy = np.mean(binary_fp == binary_ls)
                        chunk_accuracies.append(accuracy)
                        avg_accuracy += accuracy
                        num_chunks += 1
                        print(f"Accuracy for chunk {num_chunks}: {accuracy * 100:.2f}%")
    
    # Normalize predictions
    fx = np.divide(fx, weight, out=np.zeros_like(fx), where=weight != 0)
    fx = np.transpose(fx)
    
    avg_accuracy /= num_chunks

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Average accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Total time taken: {elapsed_time:.2f} seconds")    
    fx.tofile(fpath + "ChannelSeismic-DS43-probability.dat", format="%4")
    output_file = fpath + "ChannelSeismic-DS43-probability.npy"
    np.save(output_file, fx)


def loadData(n1, n2, n3, path, fname):
    gx = np.fromfile(path + fname, dtype=np.single)
    gm, gs = np.mean(gx), np.std(gx)
    gx = gx - gm
    gx = gx / gs
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
def loadDatax(n1, n2, n3, path, fname):
    gx = np.fromfile(path + fname, dtype=np.single)
    gx = np.reshape(gx, (n3, n2, n1))
    gx = np.transpose(gx)
    return gx

def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s

if __name__ == '__main__':
    main(sys.argv)