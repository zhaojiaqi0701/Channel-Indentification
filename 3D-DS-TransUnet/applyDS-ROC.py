# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# from unet3 import UNet3D  # 请确保你有这个模型定义
# from DATransUNet import DA_Transformer  # 请确保你有这个模型定义
# import configs as configs
# def main():
#     # 加载模型
#     model_unet3d = load_model_unet3d('/root/autodl-tmp/check3/checkpoint7.50.pth')
#     model_da_transunet = load_model_da_transunet("/root/autodl-tmp/check2/checkpoint6.50.pth")

#     # 使用 UNet3D 模型进行验证
#     y_true_unet3d, y_scores_unet3d = goFakeValidation_unet3d(model_unet3d)

#     # 使用 DA-TransUnet 模型进行验证
#     y_true_da_transunet, y_scores_da_transunet = goFakeValidation_da_transunet(model_da_transunet)

#     # 确保 y_true 是二进制标签
#     if torch.is_tensor(y_true_unet3d):
#         y_true_unet3d = y_true_unet3d.numpy()
#     if torch.is_tensor(y_true_da_transunet):
#         y_true_da_transunet = y_true_da_transunet.numpy()

#     y_true_unet3d = (y_true_unet3d > 0.5).astype(int)
#     y_true_da_transunet = (y_true_da_transunet > 0.5).astype(int)

#     # 绘制并对比ROC曲线
#     plt.figure(figsize=(12, 6))

#     plt.subplot(1, 2, 1)
#     plot_roc_curve(y_true_unet3d, y_scores_unet3d, "UNet3D")
#     plot_roc_curve(y_true_da_transunet, y_scores_da_transunet, "DA-TransUnet")
#     plt.title("ROC Comparison")
#     plt.legend(loc="lower right")
#     plt.savefig("roc_comparison.png")
#     plt.show()

#     # 绘制并对比Precision-Recall曲线
#     plt.figure(figsize=(12, 6))

#     plt.subplot(1, 2, 2)
#     plot_precision_recall_curve(y_true_unet3d, y_scores_unet3d, "UNet3D")
#     plot_precision_recall_curve(y_true_da_transunet, y_scores_da_transunet, "DA-TransUnet")
#     plt.title("Precision-Recall Comparison")
#     plt.legend(loc="lower left")
#     plt.savefig("pr_comparison.png")
#     plt.show()


# def load_model_unet3d(checkpoint_path):
#     model = UNet3D()
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['state_dict'])
#     model.eval()
#     return model

# def load_model_da_transunet(checkpoint_path):
#     config = configs.get_r50_b16_config()
#     model = DA_Transformer(config, img_size=128, num_classes=21843, zero_head=False, vis=False)
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['state_dict'])
#     model.eval()
#     return model

# def loadData(n1, n2, n3, path, fname):
#     gx = np.fromfile(os.path.join(path, fname), dtype=np.float32)
#     gm, gs = np.mean(gx), np.std(gx)
#     gx = (gx - gm) / gs
#     gx = np.reshape(gx, (n3, n2, n1))
#     gx = np.transpose(gx)
#     return gx

# def goFakeValidation_unet3d(model):
#     n1, n2, n3 = 256, 256, 256
#     seisPath = "/root/autodl-tmp/data/validation1/nx/"
#     lxpath = "/root/autodl-tmp/data/validation1/lx/"
#     predPath = "/root/autodl-tmp/data/validation1/px/"
#     ks = [109]
#     for k in ks:
#         fname = str(k)
#         gx = loadData(n1, n2, n3, seisPath, fname + '.dat')
#         gs = np.reshape(gx, (1, 1, n1, n2, n3))  # 修改为PyTorch格式
#         gs = torch.from_numpy(gs).float()

#         # label
#         lx = loadData(n1, n2, n3, lxpath, fname + '.dat')
#         ls = np.reshape(lx, (1, 1, n1, n2, n3))
#         ls = torch.from_numpy(ls).float()

#         with torch.no_grad():
#             fp = model(gs)
#         fp = fp[0, 0, :, :, :].cpu().numpy()
#         fp.tofile(predPath + fname + '.dat', format="%4")
#         np.save(predPath + fname + "_predictions.npy", fp)

#     return ls.flatten(), fp.flatten()

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

# def goFakeValidation_da_transunet(model):
#     n1, n2, n3 = 256, 256, 256  # 原始数据大小
#     input_size = 128  # 模型训练使用的子图像大小
#     overlap = 4  # 重叠区域大小
#     seisPath = "/root/autodl-tmp/data/validation/nx/"
#     lxpath = "/root/autodl-tmp/data/validation/lx/"
#     predPath = "/root/autodl-tmp/data/validation/px/"
#     ks = [109]

#     trilinear_weights = create_trilinear_weights(input_size, overlap)

#     for k in ks:
#         fname = str(k)
#         gx = loadData(n1, n2, n3, seisPath, fname + '.dat')  # 加载数据
#         gx = np.reshape(gx, (1, 1, n1, n2, n3))  # 转换为 PyTorch 格式
#         gx = torch.from_numpy(gx).float()
#         lx = loadData(n1, n2, n3, lxpath, fname + '.dat')  # 加载数据
#         lx = np.reshape(lx, (n1, n2, n3))  # 展平到 n1*n2*n3

#         m1, m2, m3 = input_size, input_size, input_size
#         step1 = m1 - overlap
#         step2 = m2 - overlap
#         step3 = m3 - overlap

#         num_chunks_n1 = (n1 - input_size) // step1 + 1
#         num_chunks_n2 = (n2 - input_size) // step2 + 1
#         num_chunks_n3 = (n3 - input_size) // step3 + 1

#         fx = np.zeros((n1, n2, n3), dtype=np.float32)  # 用于存储预测结果的数组
#         weight = np.zeros((n1, n2, n3), dtype=np.float32)  # 用于存储权重，用于插值

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

#                         f = model(g).cpu().numpy()  # 保持模型输出为连续的概率分布

#                         # 加权累积预测结果和权重
#                         fx[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += f[0, 0] * trilinear_weights
#                         weight[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += trilinear_weights

#             # 对所有区域进行平滑处理，确保 fx 是连续的概率分布
#             fx = np.divide(fx, weight, out=np.zeros_like(fx), where=weight != 0)

#         # 保存原始概率输出
#         fx.tofile(predPath + fname + '.dat', format="%4")
#         np.save(os.path.join(predPath, fname + "_predictions.npy"), fx)

#     return lx.flatten(), fx.flatten()

# def plot_roc_curve(y_true, y_scores, model_name):
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:0.2f})')

# def plot_precision_recall_curve(y_true, y_scores, model_name):
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     avg_precision = average_precision_score(y_true, y_scores)
#     plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_precision:0.2f})')

# if __name__ == "__main__":
#     main()
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, matthews_corrcoef
import matplotlib.pyplot as plt
from unet3 import UNet3D
import time
from DS_TransUNet3D import UNet3D
import configs as configs
def main(argv):
    load_model(argv[0])
    goFakeValidation(model)

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

def load_model(checkpoint_path):
    global model
    model = UNet3D(128, 1)
    checkpoint = torch.load("/root/autodl-tmp/Channel-checkDS/checkpointDS.43.pth")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


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

def goFakeValidation(model):
    n1, n2, n3 = 256, 256, 256
    input_size = 128
    overlap = 0
    seisPath = "/root/autodl-tmp/data/validationDS/nx/"
    lxpath = "/root/autodl-tmp/data/validationDS/lx/"
    predPath = "/root/autodl-tmp/data/validationDS/px/"
    ks = [251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]

    trilinear_weights = create_trilinear_weights(input_size, overlap)

    for k in ks:
        start_time = time.time()

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

        fx = np.zeros((n1, n2, n3), dtype=np.single)
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

                        f = model(g)

                        if isinstance(f, (tuple, list)):
                            f = f[0]

                        f = f.cpu().numpy()
                        l = l.cpu().numpy()

                        binary_fp = (f > 0.5).astype(int)
                        binary_ls = (l > 0.5).astype(int)
                        accuracy, recall, iou, dice, precision, f1_score, mcc = calculate_metrics(binary_fp, binary_ls)
                        chunk_metrics.append((accuracy, recall, iou, dice, precision, f1_score, mcc))

                        avg_accuracy += accuracy
                        avg_recall += recall
                        avg_iou += iou
                        avg_dice += dice
                        avg_precision += precision
                        avg_f1_score += f1_score
                        avg_mcc += mcc
                        num_chunks += 1
                        print(f"Metrics for chunk {num_chunks}: Accuracy={accuracy * 100:.2f}%, Recall={recall * 100:.2f}%, "
                              f"IOU={iou * 100:.2f}%, Dice={dice * 100:.2f}%, Precision={precision * 100:.2f}%, "
                              f"F1 Score={f1_score * 100:.2f}%, MCC={mcc:.2f}")

                        fx[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += f[0, 0] * trilinear_weights
                        weight[start_n1:end_n1, start_n2:end_n2, start_n3:end_n3] += trilinear_weights

            fx = np.divide(fx, weight, out=np.zeros_like(fx), where=weight != 0)

        avg_accuracy /= num_chunks
        avg_recall /= num_chunks
        avg_iou /= num_chunks
        avg_dice /= num_chunks
        avg_precision /= num_chunks
        avg_f1_score /= num_chunks
        avg_mcc /= num_chunks

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Average metrics for {fname}: Accuracy={avg_accuracy * 100:.2f}%, Recall={avg_recall * 100:.2f}%, "
              f"IOU={avg_iou * 100:.2f}%, Dice={avg_dice * 100:.2f}%, Precision={avg_precision * 100:.2f}%, "
              f"F1 Score={avg_f1_score * 100:.2f}%, MCC={avg_mcc:.2f}")
        print(f"Processing time for {fname}: {elapsed_time:.2f} seconds")

        # 绘制 ROC 和 Precision-Recall 曲线，并显示 MCC
        plot_pr_curve(lx.flatten(), fx.flatten(), fname, avg_mcc)
        plot_roc_curve(lx.flatten(), fx.flatten(), fname, avg_mcc)

def plot_roc_curve(gx, fp, fname, mcc):
    gx_binary = np.clip(np.rint(gx), 0, 1)
    fpr, tpr, thresholds = roc_curve(gx_binary, fp, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}, MCC={mcc:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'ROC_curveDS_{fname}.png')
    plt.show()

def plot_pr_curve(gx, fp, fname, mcc):
    gx_binary = np.clip(np.rint(gx), 0, 1)
    precisions, recalls, thresholds = precision_recall_curve(gx_binary, fp, pos_label=1)
    ap = average_precision_score(gx_binary, fp)

    plt.figure()
    plt.plot(recalls, precisions, label=f'AP={ap:.4f}, MCC={mcc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'PR_curveDS_{fname}.png')
    plt.show()

if __name__ == "__main__":
    main(["checkpointDS.43.pth"])  # 请替换为你的模型检查点路径

