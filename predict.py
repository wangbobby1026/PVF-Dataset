import time
import glob
import os
import time
import numpy as np
import pandas as pd
import model
import torch
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from torch import nn
from torchvision import transforms
from PIL import Image

from PV_Classify.model import effnetv2_s


def find_png_files(folder_path):
    png_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                png_files.append(os.path.join(root, file))

    return png_files


def trans(arry):
    aa = np.array(arry)
    bb = np.zeros((10, 10), dtype='int64')
    cc = np.zeros((10, 10), dtype='int64')

    for i in range(10):
        if i == 0:
            bb[i] = aa[i]
        elif i == 9:
            bb[i] = aa[1]
        else:
            bb[i] = aa[i+1]

    for j in range(10):
        if j == 0:
            cc[:,j] = bb[:,j]
        elif j == 9:
            cc[:,j] = bb[:,1]
        else:
            cc[:,j] = bb[:,j+1]
    return cc

def pre_classify_ppf(image_path, result_path, result_name, model, weights_path, resize, save):
    # img_path =''
    # 权重参数路径
    # weights_path = r''
    # 预测索引对应的类别名称
    class_names = ['6', '7', '4', '8', '9', '5', '2', '3', '1', '10']
    # class_names = ['1', '2', '3', '4', '5', '6', '2', '8', '9', '10']
    # 获取GPU设备
    if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
        device = torch.device('cuda:0')
        print('cuda:0')
    else:
        device = torch.device('cpu')
        print('cpu')

    # -------------------------------------------------- #
    # （1）数据加载
    # -------------------------------------------------- #
    # 预处理函数
    data_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor()])
    # 读取图片
    ts = time.perf_counter()
    files = find_png_files(image_path)
    # print('files', files)
    pre_result = []
    error = 0
    test_num = len(files)
    y_true = []
    y_pred = []
    # 加载模型
    model = model
    # 加载权重文件
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    # 模型切换成验证模式，dropout和bn切换形式
    for file in files:
        label = file.split('\\')[-2][:2]
        label = int(label)
        item = file.split('\\')[-2:]
        class_fold = item[0]
        fn = item[1]
        item = class_fold + '/' + fn
        frame = Image.open(file)
        # 数据预处理
        img = data_transform(frame)
        # 给图像增加batch维度 [c,h,w]==>[b,c,h,w]
        img = torch.unsqueeze(img, dim=0)
        # 前向传播过程中不计算梯度
        with torch.no_grad():
            # 前向传播
            outputs = model(img)
            # 只有一张图就挤压掉batch维度
            outputs = torch.squeeze(outputs)
            # 计算图片属于7个类别的概率
            predict = torch.softmax(outputs, dim=0)
            # 得到类别索引
            predict_cla = torch.argmax(predict).numpy()
        # 获取最大预测类别概率
        predict_score = round(torch.max(predict).item(), 4)
        # 获取预测类别的名称
        predict_name = class_names[predict_cla]
        print(label)
        preclass = int(predict_name)
        print(predict_name)
        if label != preclass:
            error += 1
            print('error')
        print([item, predict_name, predict_score])
        pre_result.append([item, predict_name, predict_score])
        y_true.append(str(label))
        y_pred.append(predict_name)

    haha = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    haha.to_csv('111.csv', index=False)

    acc = error / test_num
    # print(pre_result)
    print('error num:{}, accuracy:{}'.format(error, 1 - acc))
    if save == True:
        column = ['image', 'class', 'score']  # 列表对应每列的列名
        test = pd.DataFrame(columns=column, data=pre_result)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_csv = result_path + '/{}.csv'.format(result_name)
        # print(result_csv)
        test.to_csv(result_csv, encoding='ANSI')  # 如果生成excel，可以用to_excel
    else:
        pass
    confusion_mat = confusion_matrix(y_true, y_pred)
    confusion_mat = trans(confusion_mat)
    print(confusion_mat)
    observed = []
    predicted = []
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat)):
            observed.extend([i] * confusion_mat[i][j])
            predicted.extend([j] * confusion_mat[i][j])
    # 计算 Cohen's Kappa 系数
    kappa_score = cohen_kappa_score(observed, predicted)
    print("Cohen's Kappa 系数：", kappa_score)
    te = time.perf_counter()
    times = te - ts
    log_list = []
    log_list.append('测试样本数：{}  错误样本数：{}  acc：{}'.format(test_num, error, 1 - acc))
    log_list.append(confusion_mat)
    log_list.append(("Cohen's Kappa 系数：{}".format(kappa_score)))
    log_list.append('预测速度：{} fps'.format(test_num / times))
    log_list.append('用时：{}'.format(times))
    if save == True:
        result_log = os.path.join(result_path, '{}.txt'.format(result_name))
        with open(result_log, 'a', encoding='utf-8') as f:  # 使用'a'模式打开文件，可以追加写入
            # 将本次迭代的log信息添加到log列表中
            f.write('{}'.format(log_list))
            # f.write('\n'.join(log))  # 将log列表中的信息写入文件，每行末尾添加换行符
            f.write('\n')  # 最后添加一行空行


if __name__ == '__main__':
    image_path = r'C:\Users\Wbobby\Desktop\dataset\PPFdataset_10\test2'
    # cl_model = model.create_model('resnet50', num_classes=10)
    cl_model = effnetv2_s()
    weights_path = r'D:\PV\PVFdataset\experiment110x60\effnetv2_s_train2024421523\epoch206.pth'
    result_path = os.path.split(weights_path)[0]
    print(result_path)
    result_name = os.path.splitext(os.path.basename(weights_path))[0]
    t0 = time.perf_counter()
    pre_classify_ppf(image_path, result_path, result_name, cl_model, weights_path, resize=224, save=False)
    t1 = time.perf_counter()
    print('识别时间：{}'.format(t1 - t0))
