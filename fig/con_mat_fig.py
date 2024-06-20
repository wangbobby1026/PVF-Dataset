"""
@FileName：con_mat_fig.py\n
@Description：\n
@Author：WBobby\n
@Department：CUG\n
@Time：2024/3/24 13:07\n
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def ax_fig(ax, con_mat, sub_title, labels):
    num_classes = con_mat.shape[0]
    # 生成自定义的颜色数组
    # colors = plt.cm.Wistia()
    #
    # # 创建自定义的颜色映射
    # custom_cmap = ListedColormap(colors)
    im = ax.imshow(con_mat, interpolation='nearest', cmap=plt.cm.Wistia)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, int(con_mat[i, j]), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='black' if con_mat[i, j] > 0 else 'black')
    ax.set_title(sub_title, fontsize=17)
    # ax.set_xlabel('Predicted label', fontsize=13)
    # ax.set_ylabel('True label', fontsize=20)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticklabels([])
    return im


def con_mat_fig(con_mats, title_arr, out_fig):
    plt.rc('font', family='Times New Roman')
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    ax5 = axes[4]
    ax = (ax1, ax2, ax3, ax4, ax5)
    sub_img = []
    for i, con_mat in enumerate(con_mats):
        im = ax_fig(ax[i], con_mat, sub_title=title_arr[i], labels=labels)
        sub_img.append(im)
        ax1.set_yticklabels(labels, fontsize=13)
        ax1.set_ylabel('True label', fontsize=13)
        ax3.set_xlabel('Predicted label', fontsize=13)
    cbar = fig.colorbar(sub_img[-1], ax=axes, fraction=0.01, pad=0.01)
    plt.show()
    # plt.savefig(out_fig, dpi=500)


if __name__ == '__main__':
    title_arr = ['Res50', 'Effv2-s', 'ViT-s', 'Swinv2-t', 'Coat-ls']
    con_matres = np.array([[59, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 6, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 31, 0, 0, 0, 0, 2, 8],
                           [0, 0, 1, 0, 26, 1, 0, 0, 2, 0],
                           [0, 0, 0, 0, 0, 30, 0, 0, 2, 5],
                           [0, 0, 0, 0, 0, 0, 10, 0, 3, 0],
                           [0, 0, 0, 0, 0, 0, 0, 77, 1, 2],
                           [0, 0, 0, 1, 1, 4, 0, 1, 84, 3],
                           [0, 0, 0, 3, 0, 5, 0, 1, 2, 140]], dtype='int64')

    con_mateff = np.array([[58, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 34, 0, 0, 0, 2, 1, 4],
                           [0, 0, 0, 0, 28, 0, 0, 0, 2, 0],
                           [0, 0, 0, 0, 1, 31, 0, 0, 1, 4],
                           [0, 0, 0, 0, 1, 0, 9, 0, 3, 0],
                           [0, 0, 0, 0, 0, 0, 0, 77, 1, 2],
                           [0, 0, 2, 0, 1, 2, 0, 1, 83, 5],
                           [0, 0, 0, 4, 0, 3, 0, 0, 1, 143]], dtype='int64')
    con_matvit = np.array([[53, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                           [0, 41, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 3, 3, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 14, 0, 0, 0, 4, 4, 19],
                           [0, 1, 0, 0, 22, 0, 0, 1, 5, 1],
                           [0, 0, 0, 3, 2, 26, 0, 0, 1, 5],
                           [0, 0, 2, 0, 1, 0, 9, 0, 1, 0],
                           [0, 0, 0, 3, 0, 1, 0, 72, 3, 1],
                           [0, 2, 2, 1, 1, 5, 0, 3, 76, 4],
                           [1, 0, 0, 10, 0, 2, 0, 2, 2, 134]], dtype='int64')
    con_matswin = np.array([[56, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                            [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 6, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 32, 0, 0, 0, 0, 3, 6],
                            [0, 0, 0, 1, 26, 0, 0, 0, 3, 0],
                            [0, 0, 0, 0, 0, 32, 0, 0, 0, 5],
                            [0, 2, 0, 0, 1, 0, 8, 0, 2, 0],
                            [0, 0, 0, 0, 0, 1, 0, 76, 1, 2],
                            [0, 0, 1, 2, 0, 3, 0, 0, 87, 1],
                            [0, 0, 0, 10, 0, 2, 0, 0, 2, 137]], dtype='int64')
    con_matcoat = np.array([[58, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 5, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 31, 0, 0, 0, 1, 3, 6],
                            [0, 0, 0, 1, 25, 0, 0, 1, 3, 0],
                            [0, 0, 0, 0, 0, 30, 0, 0, 3, 4],
                            [0, 0, 0, 0, 1, 0, 10, 1, 1, 0],
                            [0, 0, 0, 0, 0, 1, 0, 77, 0, 2],
                            [0, 0, 1, 0, 2, 1, 0, 1, 86, 3],
                            [0, 0, 0, 2, 0, 3, 0, 1, 1, 144]], dtype='int64')

    con_mats = [con_matres, con_mateff, con_matvit, con_matswin, con_matcoat]
    out_fig = 'conmat.png'
    con_mat_fig(con_mats, title_arr,out_fig)
