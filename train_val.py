"""
@FileName：train_val.py\n
@Description：用于训练及其验证\n
@Author：WBobby\n
@Department：CUG\n
@Time：2023/4/28 23:21\n
"""

import argparse
import os
import time
import model
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torch
from ResNET.my_dataset import MyDataSet
from ResNET.utils import read_split_data


def train_val(args, model, model_param):
    ts = time.perf_counter()
    t = time.localtime()
    information = '{}-{}-{} {}:{}:{}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
    log = []
    log.append(
        '训练时间：{}, batch_size：{}, epochs：{}, lr:{}'.format(information, args.batch_size, args.epochs, args.lr))
    save_path = os.path.join(args.save_path,
                             '{}_train{}{}{}{}{}{}'.format(args.model_name, t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour,
                                                           t.tm_min, t.tm_sec))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 如果有GPU就用，没有就用CPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('device: CUDA:0')
    else:
        device = torch.device('cpu')
        print('device: CPU')

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 图片预处理

    transform_train = transforms.Compose([
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(180, expand=True, center=None, fill=0),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(args.input_size),
        transforms.ToTensor()])

    transform_val = transforms.Compose([
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(180, expand=False, center=None, fill=0),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(args.input_size),
        transforms.ToTensor()])

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=transform_train)
    train_num = len(train_images_path)

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=transform_val)
    val_num = len(val_images_path)

    log_file = os.path.join(save_path, 'log.txt')
    with open(log_file, 'a', encoding='utf-8') as f:  # 使用'a'模式打开文件，可以追加写入
        f.write('{}\n'.format(model_param))
        f.write('{}images for training \n{}images for validation\n'.format(train_num, val_num))
        f.write('\n')
    batch_size = args.batch_size

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0)  # 他说win得是0 我要试试；我信了只能为0

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)  # 加载数据时的线程数量，windows环境下只能=0

    net = model
    net.to(device)
    loss_function = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0005)  # 定义优化器，设置学习率

   # 定义学习率衰减策略
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    epochs = args.epochs  # 迭代次数
    best_val = 0.0
    print('开始训练')
    for epoch in range(epochs):
        print('-' * 30, '\n', 'epoch:', epoch)
        net.train()
        running_loss = 0.0
        t1 = time.perf_counter()
        '''
        enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中,
        start 表示下标
        '''
        for step, data in enumerate(train_data_loader):
            # data中包含图像及其对应的标签
            images, labels = data
            # 梯度清零，因为每次计算梯度是一个累加
            optimizer.zero_grad()
            # 前向传播
            outputs = net(images.to(device))
            # 计算预测值和真实值的交叉熵损失
            loss = loss_function(outputs, labels.to(device))
            # 梯度计算,反向传播
            loss.backward()
            # 权重更新
            optimizer.step()
            # 累加每个step的损失
            running_loss += loss.item()
            rate = (step + 1) / len(train_data_loader)
            print(f'rate:{rate} loss:{loss}')
            
        scheduler.step()
        lr = scheduler.get_last_lr()
        print("lr: {}".format(lr))
        
        # val
        net.eval()  # 切换为验证模型，BN和Dropout不起作用
        acc = 0.0  # 验证集准确率
        with torch.no_grad():  # 下面不进行梯度计算
            for data_var in val_data_loader:
                # 获取验证集的图片和标签
                var_images, var_labels = data_var
                # 前向传播
                outputs = net(var_images.to(device))
                # 求取最有可能的预测类别
                predict_y = torch.max(outputs, dim=1)[1]
                # 累加每个step的准确率
                acc += (predict_y == var_labels.to(device)).sum().item()
            accurate_val = acc / val_num
            if accurate_val >= best_val:
                best_val = accurate_val
                torch.save(net.state_dict(), os.path.join(save_path, 'epoch{}.pth'.format(epoch)))
            if epoch == epochs - 1:
                torch.save(net.state_dict(), os.path.join(save_path, 'epoch{}.pth'.format(epoch)))
            print('finish epoch:{}, acc:{}, best_val:{}, loss: {}, time:{}'.format(
                epoch, accurate_val, best_val, running_loss / (step + 1), time.perf_counter() - t1))
            print('save: epoch{}.pth'.format(epoch))
            with open(log_file, 'a', encoding='utf-8') as f:  # 使用'a'模式打开文件，可以追加写入
                message = 'finish epoch:{}, acc:{}, loss: {}, time:{}'.format(epoch, accurate_val,
                                                                              running_loss / (step + 1),
                                                                              time.perf_counter() - t1)
                f.write('{}\n'.format(message))  # 将log列表中的信息写入文件，每行末尾添加换行符
    print('best_val:', best_val)

    with open(log_file, 'a', encoding='utf-8') as f:
        tend = time.perf_counter()
        te = time.localtime()
        t_end = '{}-{}-{} {}:{}:{}'.format(te.tm_year, te.tm_mon, te.tm_mday, te.tm_hour, te.tm_min, te.tm_sec)
        f.write('训练完成！！！！   best_acc: {} 时间：{}\n训练时长：{}s'.format(best_val, t_end,
                                                                           tend - ts))  # 将log列表中的信息写入文件，每行末尾添加换行符
        f.write('\n')  # 最后添加一行空行


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'C:\Users\Wbobby\Desktop\dataset\PPFdataset112x112\train')
    parser.add_argument('--save_path', type=str, default=r'C:\Users\Wbobby\Desktop\dataset\PPFdataset112x112\result0617')
    parser.add_argument('--class_num', type=int, default=10, help='class_num')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--epochs', type=int, default='2', help='epochs')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', help='model')
    parser.add_argument('--input_size', type=tuple, default=(224, 224), help='model')
    args = parser.parse_args()
    t0 = time.perf_counter()
    model_name_list = ['resnet50', 'efficientnet_b0', 'vit_small_patch32_224_in21k', 'coat_lite_small', 'swinv2_tiny_window7_224']
    lr = [0.0005, 0.0005, 0.0002, 0.0002, 0.0005]
    for i, model_name in enumerate(model_name_list):
        args.model_name = model_name
        args.lr = lr[i]
        model_param = {'model_name:': args.model_name, 'input_size': args.input_size, 'batch_size': args.batch_size,
                       'lr': args.lr, 'epochs': args.epochs}
        cl_model = model.create_model(args.model_name, pretrained=False, num_classes=args.class_num)
        train_val(args, cl_model, model_param)
        t1 = time.perf_counter()
        print('finish train !!!!!', 'time: {}'.format(t1 - t0))
