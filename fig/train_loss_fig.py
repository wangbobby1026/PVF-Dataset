import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def csv2dict(csv_file):
    data = pd.read_csv(csv_file)
    dict = {}
    for i in range(len(data)):
        if i % 5 == 0:
            epoch = data.iloc[i, 1] + 1
            val_acc = data.iloc[i, 3]
            loss = data.iloc[i, 5]
            dict[epoch] = [val_acc, loss]
    return dict


def loss_fig(csv_path):
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']
    figsize = (16, 9)
    plt.figure(figsize=figsize)
    csv_files = os.listdir(csv_path)
    csv_names = [os.path.join(csv_path, csv_file) for csv_file in csv_files]
    model_names = ['Coat-ls', 'Effv2-s', 'Res-50', 'Swinv2-t', 'ViT-s']
    model_losscolors = plt.cm.coolwarm([0, 0.1, 0.7, 0.8, 0.9])
    model_valcolors = plt.cm.coolwarm([0.02, 0.12, 0.68, 0.78, 0.88])
    line_styles = ['-', '--', '-.', ':', '--']
    # linewidths = [2, 1.8, 1.6, 1.4, 1.4]
    linewidths = [2, 2, 2, 2, 2]
    # alphas = [1, 0.8, 0.7, 0.6, 0.6]
    alphas = [1, 1, 1, 1, 1]
    model_losscolors = ['lightcoral', 'orange', 'lightgreen', 'darkturquoise', 'plum']
    model_valcolors = ['indianred', 'darkorange', 'mediumseagreen', 'c', 'orchid']
    markers = ['o', 'v', '^', 's', '*']
    ax1 = plt.subplot()
    ax2 = plt.twinx()
    zz = zip(model_names, csv_names, model_losscolors, model_valcolors, line_styles, linewidths, alphas, markers)

    lines = []
    labels = []
    lines_labels = []
    for model_name, csv_name, model_color, model_valcolor, line_style, linewidth, alpha, markers in zz:
        csv_file = os.path.join(csv_path, csv_name)
        dict = csv2dict(csv_file)
        epochs = list(dict.keys())
        val_acc = [item[0] for item in dict.values()]
        loss = [item[1] for item in dict.values()]
        loss_line, = ax1.plot(epochs, loss, color=model_color, linestyle=line_style, linewidth=linewidth, alpha=alpha,
                             marker=markers,
                             markersize=5, label='Loss {}'.format(model_name))
        val_line, = ax2.plot(epochs, val_acc, color=model_valcolor, marker=markers, markersize=5, linestyle=line_style,
                            alpha=alpha,
                            linewidth=linewidth, label='Val Acc {}'.format(model_name))
        lines.append(val_line)
        lines.append(loss_line)
        labels.append(val_line.get_label())
        labels.append(loss_line.get_label())

    # 将线条和标签的元组列表解包，并传入图例中


    # 创建统一的图例
    plt.legend(lines, labels, loc='center right', fontsize=17)
    plt.title('Model Loss and Validation Accuracy', fontsize=40, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=25, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=20)
    ax1.set_ylabel('Train Loss', fontsize=25, fontweight='bold')
    ax1.yaxis.set_tick_params(labelsize=20)
    ax2.set_ylabel('Validation Accuracy', fontsize=25, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=20)
    # plt.legend(loc='center right', fontsize=20)
    plt.savefig(r'C:\Users\Wbobby\Documents\TeX_files\PVFC\tu/' + 'loss_val1.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    csv_parth = r'C:\Users\Wbobby\Desktop\csv'
    out = ''
    loss_fig(csv_parth)
