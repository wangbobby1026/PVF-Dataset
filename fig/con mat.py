"""
@FileName：con mat.py\n
@Description：\n
@Author：WBobby\n
@Department：CUG\n
@Time：2024/4/2 20:46\n
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

from fig.con_mat_fig import con_mat_fig

'''ori panel images'''

con_matres_ori = np.array([[56, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                           [0, 40, 1, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 6, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 1, 31, 0, 0, 0, 0, 2, 7],
                           [0, 0, 2, 0, 25, 0, 1, 0, 2, 0],
                           [0, 0, 0, 0, 0, 29, 0, 0, 4, 4],
                           [0, 0, 1, 0, 0, 0, 9, 0, 3, 0],
                           [0, 0, 0, 0, 0, 0, 0, 77, 1, 2],
                           [0, 0, 1, 1, 0, 2, 0, 1, 87, 2],
                           [1, 0, 0, 2, 0, 1, 0, 0, 0, 147]], dtype='int64')

con_mateff_ori = np.array([[56, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                           [0, 40, 1, 0, 0, 0, 0, 0, 1, 0],
                           [0, 1, 5, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 34, 1, 0, 0, 2, 1, 3],
                           [0, 0, 2, 0, 24, 1, 0, 0, 3, 0],
                           [0, 0, 0, 0, 0, 29, 1, 0, 3, 4],
                           [0, 1, 0, 0, 0, 0, 11, 0, 1, 0],
                           [0, 0, 0, 1, 0, 0, 0, 77, 0, 2],
                           [0, 0, 1, 2, 0, 0, 1, 1, 87, 2],
                           [0, 0, 0, 5, 0, 4, 0, 1, 1, 140]], dtype='int64')
con_matvit_ori = np.array([[54, 1, 0, 0, 1, 1, 1, 0, 0, 1],
                           [0, 41, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 2, 1, 2, 0],
                           [0, 0, 0, 17, 0, 0, 0, 0, 6, 18],
                           [0, 0, 0, 1, 23, 1, 0, 1, 2, 2],
                           [1, 0, 1, 0, 1, 26, 0, 1, 4, 3],
                           [3, 2, 0, 0, 0, 0, 8, 0, 0, 0],
                           [1, 0, 0, 0, 0, 1, 0, 64, 2, 12],
                           [0, 1, 1, 3, 0, 4, 2, 3, 63, 17],
                           [0, 0, 0, 10, 0, 2, 0, 7, 2, 130]], dtype='int64')
con_matswin_ori = np.array([[56, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                            [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 4, 0, 0, 0, 1, 0, 2, 0],
                            [1, 0, 0, 22, 1, 0, 0, 0, 3, 14],
                            [0, 0, 0, 0, 23, 1, 1, 1, 4, 0],
                            [0, 0, 0, 0, 0, 30, 0, 0, 4, 3],
                            [0, 0, 0, 0, 0, 0, 10, 0, 3, 0],
                            [0, 0, 0, 1, 0, 0, 0, 77, 1, 1],
                            [0, 0, 0, 3, 0, 0, 2, 0, 88, 1],
                            [0, 0, 0, 2, 0, 1, 0, 0, 6, 142]], dtype='int64')
con_matcoat_ori = np.array([[58, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 4, 0, 0, 0, 0, 0, 2, 0],
                            [0, 0, 0, 31, 0, 0, 0, 1, 2, 7],
                            [0, 1, 1, 1, 22, 0, 0, 0, 5, 0],
                            [0, 0, 0, 0, 0, 31, 0, 0, 4, 2],
                            [0, 0, 0, 0, 0, 0, 8, 0, 5, 0],
                            [0, 0, 0, 0, 0, 0, 0, 76, 2, 2],
                            [0, 0, 0, 0, 0, 2, 0, 1, 89, 2],
                            [0, 0, 0, 4, 0, 2, 0, 1, 1, 143]], dtype='int64')

'''resample'''

con_matres_samp = np.array([[59, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 6, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 31, 0, 0, 0, 0, 2, 8],
                            [0, 0, 1, 0, 26, 1, 0, 0, 2, 0],
                            [0, 0, 0, 0, 0, 30, 0, 0, 2, 5],
                            [0, 0, 0, 0, 0, 0, 10, 0, 3, 0],
                            [0, 0, 0, 0, 0, 0, 0, 77, 1, 2],
                            [0, 0, 0, 1, 1, 4, 0, 1, 84, 3],
                            [0, 0, 0, 3, 0, 5, 0, 1, 2, 140]], dtype='int64')

con_mateff_samp = np.array([[58, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 34, 0, 0, 0, 2, 1, 4],
                            [0, 0, 0, 0, 28, 0, 0, 0, 2, 0],
                            [0, 0, 0, 0, 1, 31, 0, 0, 1, 4],
                            [0, 0, 0, 0, 1, 0, 9, 0, 3, 0],
                            [0, 0, 0, 0, 0, 0, 0, 77, 1, 2],
                            [0, 0, 2, 0, 1, 2, 0, 1, 83, 5],
                            [0, 0, 0, 4, 0, 3, 0, 0, 1, 143]], dtype='int64')
con_matvit_samp = np.array([[53, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                            [0, 41, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 3, 3, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 14, 0, 0, 0, 4, 4, 19],
                            [0, 1, 0, 0, 22, 0, 0, 1, 5, 1],
                            [0, 0, 0, 3, 2, 26, 0, 0, 1, 5],
                            [0, 0, 2, 0, 1, 0, 9, 0, 1, 0],
                            [0, 0, 0, 3, 0, 1, 0, 72, 3, 1],
                            [0, 2, 2, 1, 1, 5, 0, 3, 76, 4],
                            [1, 0, 0, 10, 0, 2, 0, 2, 2, 134]], dtype='int64')
con_matswin_samp = np.array([[56, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                             [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 6, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 32, 0, 0, 0, 0, 3, 6],
                             [0, 0, 0, 1, 26, 0, 0, 0, 3, 0],
                             [0, 0, 0, 0, 0, 32, 0, 0, 0, 5],
                             [0, 2, 0, 0, 1, 0, 8, 0, 2, 0],
                             [0, 0, 0, 0, 0, 1, 0, 76, 1, 2],
                             [0, 0, 1, 2, 0, 3, 0, 0, 87, 1],
                             [0, 0, 0, 10, 0, 2, 0, 0, 2, 137]], dtype='int64')
con_matcoat_samp = np.array([[58, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 5, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 31, 0, 0, 0, 1, 3, 6],
                             [0, 0, 0, 1, 25, 0, 0, 1, 3, 0],
                             [0, 0, 0, 0, 0, 30, 0, 0, 3, 4],
                             [0, 0, 0, 0, 1, 0, 10, 1, 1, 0],
                             [0, 0, 0, 0, 0, 1, 0, 77, 0, 2],
                             [0, 0, 1, 0, 2, 1, 0, 1, 86, 3],
                             [0, 0, 0, 2, 0, 3, 0, 1, 1, 144]], dtype='int64')

'''resample&padding'''

con_matres_pad = np.array([[58, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 40, 0, 0, 1, 0, 1, 0, 0, 0],
                           [0, 1, 5, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 33, 0, 0, 1, 1, 0, 6],
                           [0, 0, 2, 0, 26, 0, 1, 0, 1, 0],
                           [0, 1, 0, 0, 0, 34, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0, 0, 11, 0, 2, 0],
                           [0, 0, 0, 1, 0, 0, 0, 76, 1, 2],
                           [0, 0, 0, 1, 0, 2, 0, 0, 87, 4],
                           [0, 0, 0, 2, 0, 2, 0, 3, 2, 142]], dtype='int64')

con_mateff_pad = np.array([[59, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 40, 0, 0, 1, 0, 1, 0, 0, 0],
                           [0, 1, 4, 0, 0, 0, 0, 0, 2, 0],
                           [0, 0, 0, 34, 0, 0, 1, 1, 1, 4],
                           [0, 0, 0, 0, 30, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 34, 0, 0, 2, 1],
                           [0, 0, 0, 0, 0, 0, 12, 0, 1, 0],
                           [0, 0, 0, 1, 0, 0, 0, 74, 1, 4],
                           [0, 0, 1, 2, 0, 3, 0, 0, 87, 1],
                           [0, 0, 0, 2, 0, 2, 0, 3, 2, 142]], dtype='int64')
con_matvit_pad = np.array([[58, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 38, 1, 0, 2, 0, 0, 0, 1, 0],
                           [1, 1, 2, 0, 0, 0, 1, 0, 2, 0],
                           [0, 1, 0, 16, 0, 0, 0, 2, 5, 17],
                           [0, 0, 0, 1, 24, 0, 0, 0, 3, 2],
                           [0, 0, 0, 1, 0, 29, 1, 0, 5, 1],
                           [0, 0, 0, 0, 1, 0, 11, 0, 1, 0],
                           [0, 0, 0, 2, 0, 0, 0, 74, 0, 4],
                           [0, 1, 2, 2, 0, 1, 2, 1, 78, 7],
                           [1, 0, 0, 10, 0, 0, 0, 4, 3, 133]], dtype='int64')
con_matswin_pad = np.array([[59, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 40, 0, 0, 1, 0, 1, 0, 0, 0],
                            [1, 1, 4, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 27, 0, 0, 1, 3, 1, 9],
                            [1, 0, 0, 0, 29, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 32, 0, 0, 5, 0],
                            [1, 0, 0, 0, 1, 0, 9, 0, 2, 0],
                            [0, 0, 0, 1, 0, 0, 0, 76, 0, 3],
                            [0, 0, 2, 2, 0, 1, 2, 2, 83, 2],
                            [1, 0, 0, 3, 0, 4, 0, 3, 2, 138]], dtype='int64')
con_matcoat_pad = np.array([[59, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 41, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 3, 0, 0, 0, 0, 0, 2, 0],
                            [0, 0, 0, 33, 0, 0, 1, 1, 2, 4],
                            [0, 0, 1, 0, 28, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 34, 0, 0, 3, 0],
                            [0, 0, 0, 0, 1, 0, 11, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0, 0, 75, 0, 4],
                            [0, 0, 0, 1, 0, 0, 1, 0, 89, 3],
                            [0, 0, 0, 3, 0, 2, 0, 2, 0, 144]], dtype='int64')

pvfcnetsa = np.array([[59, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 2, 3, 0, 0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 32, 0, 0, 1, 2, 0, 6],
                      [0, 0, 3, 1, 24, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 31, 0, 0, 5, 1],
                      [0, 0, 0, 0, 1, 0, 10, 0, 2, 0],
                      [0, 0, 0, 1, 0, 0, 0, 77, 1, 1],
                      [0, 0, 1, 2, 0, 1, 0, 1, 86, 3],
                      [0, 0, 0, 3, 0, 4, 0, 2, 1, 141]], dtype='int64')

pvfcnetls22 = np.array([[59, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 3, 0, 0, 0, 1, 0, 2, 0],
                        [0, 0, 0, 33, 0, 0, 2, 0, 0, 6],
                        [0, 0, 1, 0, 28, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 34, 0, 0, 1, 2],
                        [0, 0, 0, 0, 0, 0, 11, 0, 2, 0],
                        [0, 0, 0, 1, 0, 0, 0, 76, 1, 2],
                        [0, 0, 1, 2, 1, 2, 2, 0, 82, 4],
                        [0, 0, 0, 2, 0, 1, 0, 2, 2, 144]], dtype='int64')

pvfcnet22 = np.array([[59, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 41, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 1, 3, 0, 0, 0, 0, 0, 3, 0],
                      [0, 0, 0, 35, 0, 0, 1, 1, 0, 4],
                      [0, 0, 3, 0, 26, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 34, 0, 0, 3, 0],
                      [0, 0, 0, 0, 0, 0, 12, 0, 1, 0],
                      [0, 0, 0, 1, 0, 0, 0, 77, 1, 1],
                      [0, 0, 0, 1, 0, 1, 0, 1, 89, 2],
                      [0, 0, 0, 1, 0, 1, 0, 2, 2, 145]], dtype='int64')

Lk_contri = pvfcnet22-pvfcnetsa
print(Lk_contri)
Att_contri = pvfcnet22-pvfcnetls22
print(Att_contri)


# ressamp_ori = con_matres_samp - con_matres_ori
# print(ressamp_ori)
# respad_ori = con_matres_pad - con_matres_ori
# print(respad_ori)
#
# effsamp_ori = con_mateff_samp - con_mateff_ori
# print(effsamp_ori)
# effpad_ori = con_mateff_pad - con_mateff_ori
# print(effpad_ori)
#
# vitsamp_ori = con_matvit_samp - con_matvit_ori
# print(vitsamp_ori)
# vitpad_ori = con_matvit_pad - con_matvit_ori
# print(vitpad_ori)
#
# swinsamp_ori = con_matswin_samp - con_matswin_ori
# print(swinsamp_ori)
# swinpad_ori = con_matswin_pad - con_matswin_ori
# print(swinpad_ori)
#
# coatsamp_ori = con_matcoat_samp - con_matcoat_ori
# print(coatsamp_ori)
# coatpad_ori = con_matcoat_pad - con_matcoat_ori
# print(coatpad_ori)

def acc_prec_rec_f1(con_matres_ori):
    confusion_matrix = con_matres_ori
    # 初始化y_true和y_pred列表
    y_true = []
    y_pred = []

    # 遍历混淆矩阵的每个单元格
    for i in range(confusion_matrix.shape[0]):  # 行，即真实标签
        for j in range(confusion_matrix.shape[1]):  # 列，即预测标签
            # 对于每个非零单元格，添加相应的真实标签和预测标签
            for _ in range(confusion_matrix[i, j]):
                y_true.append(i)
                y_pred.append(j)

    # 打印结果
    # print("y_true:", y_true)
    # print("y_pred:", y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(accuracy_score(y_true, y_pred))
    # print(precision_score(y_true, y_pred, average='macro'))
    # print(recall_score(y_true, y_pred, average='macro'))
    # print(f1_score(y_true, y_pred, average='macro'))

    print('& {:.4f} & {:.4f} & {:.4f}'.format(prec, rec, f1))


def ax_fig(ax, con_mat, sub_title, labels, norm, cmap):
    num_classes = con_mat.shape[0]
    im = ax.imshow(con_mat, interpolation='nearest', cmap=cmap, norm=norm)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='darksalmon', lw=2))
            ax.text(j, i, int(con_mat[i, j]), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='black' if con_mat[i, j] > 0 else 'black')
    ax.set_title(sub_title, fontsize=17)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticklabels([])
    return im


def con_mat_fig(con_mats, title_arr, out_fig):
    plt.rc('font', family='Times New Roman')
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axes
    ax = (ax1, ax2, ax3, ax4, ax5, ax6)
    sub_img = []
    vmax = 144
    vmin = 0
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Choose the range of colors you want from the rainbow colormap
    start_color = 0  # Adjust these values to select your desired range
    end_color = 1
    cmap = plt.cm.Wistia(np.linspace(start_color, end_color, 256))
    cmap = ListedColormap(cmap)

    for i, con_mat in enumerate(con_mats):
        im = ax_fig(ax[i], con_mat, sub_title=title_arr[i], labels=labels, norm=norm, cmap=cmap)
        sub_img.append(im)
        ax1.set_yticklabels(labels, fontsize=13)
        ax1.set_ylabel('True label', fontsize=13)
        ax3.set_yticklabels(labels, fontsize=13)
        ax3.set_ylabel('True label', fontsize=13)
        ax5.set_xlabel('Predicted label', fontsize=13)
    cbar = fig.colorbar(sub_img[-1], ax=axes, fraction=0.01, pad=0.01)
    # plt.show()
    plt.savefig(out_fig, dpi=500)


con_mats_ori = [con_matres_ori, con_mateff_ori, con_matvit_ori, con_matswin_ori, con_matcoat_ori]
con_mats_samp = [con_matres_samp, con_mateff_samp, con_matvit_samp, con_matswin_samp, con_matcoat_samp]
con_mats_pad = [con_matres_pad, con_mateff_pad, con_matvit_pad, con_matswin_pad, con_matcoat_pad, pvfcnet22]
# con_mats_samp_ori = [ressamp_ori, effsamp_ori, vitsamp_ori, swinsamp_ori, coatsamp_ori]
# con_mats_pad_ori = [respad_ori, effpad_ori, vitpad_ori, swinpad_ori, coatpad_ori]

con_mats = [Lk_contri, Att_contri]

if __name__ == '__main__':
    title_arr = ['Res50', 'Effv2-s', 'ViT-s', 'Swinv2-t', 'Coat-ls','PVFCNet22']
    out_fig = 'con_mat_compare_samp_pad531.png'
    con_mat_fig(con_mats_pad, title_arr, out_fig)

    acc_prec_rec_f1(con_mateff_pad)

