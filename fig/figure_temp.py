"""
@FileName：figure_temp.py\n
@Description：python画图模板\n
@Author：WBobby\n
@Department：CUG\n
@Time：2024/3/20 20:29\n
"""


import matplotlib.pyplot as plt
import numpy as np

def plot_figure(feature1, feature2, filename):
    # 创建整块画布
    '''
    plt.figure() 是 matplotlib.pyplot 模块中的一个函数，用于创建一个新的图表画布（Figure对象）。这个函数有几个参数，可以用来配置图表的外观和行为。以下是一些常用的参数：
    figsize：一个元组，包含两个数值，分别代表图表的宽度和高度（以英寸为单位）。例如，figsize=(8, 6) 将创建一个宽8英寸和高6英寸的图表。
    dpi：图表的分辨率，表示每英寸多少个像素。默认值通常是 100。
    facecolor：图表画布的背景颜色。可以是一个颜色名、十六进制颜色码、RGB 或 RGBA 元组，或者预定义的颜色字符串。
    edgecolor：图表画布的边缘颜色。
    frameon：布尔值，表示是否在图表画布周围显示边框。默认值为 True。
    fig：可选的参数，用于指定要修改的现有Figure对象。如果没有提供，将创建一个新的Figure对象。
    constrained_layout：布尔值，表示是否使用受约束的布局。如果设置为 True，matplotlib 将尝试优化子图之间的空间分配。
    clear：布尔值，表示是否在创建新图表之前清除当前图表。默认值为 False。
    :param feature1:
    :param feature2:
    :return:
    '''
    fig = plt.figure(figsize=(8, 6))
    # 创建子图、子图实在画布上排布的，并且会覆盖
    '''
    在 matplotlib 中，Axes 类是用于在 Figure 对象上绘制图表和图形的核心类。每个 Axes 对象代表图表中的一个坐标轴系统，可以用来绘制线条、散点图、柱状图、饼图等多种类型的图形。
    Axes 类提供了一系列的方法和属性来控制图形的绘制和样式，以及进行数据处理和分析。以下是一些 Axes 类的主要属性和方法：
    属性:
        xaxis 和 yaxis：分别代表 x 轴和 y 轴的对象，可以用来设置轴的属性，如标签、刻度、范围等。
        lines：一个 LineCollection 对象，包含了当前 Axes 对象上的所有线条。
        collections：一个 Collection 对象，包含了当前 Axes 对象上的所有集合图形（如多边形、散点图等）。
        patches：一个 PatchCollection 对象，包含了当前 Axes 对象上的所有填充区域。
        images：一个 ImageCollection 对象，包含了当前 Axes 对象上的所有图像。
        legend_：一个 Legend 对象，包含了当前 Axes 对象的图例。
    方法:
        plot：绘制线条，接受 x 和 y 坐标，以及其他可选参数来控制线条的样式。
        scatter：绘制散点图，接受 x 和 y 坐标，以及其他可选参数来控制标记的样式。
        bar 和 barh：绘制柱状图，接受 x 和 y 坐标，以及其他可选参数来控制柱状图的样式。
        boxplot：绘制箱线图，接受数据集，以及其他可选参数来控制箱线图的样式。
        fill_between：填充两个数据集之间的区域。
        errorbar：绘制带有误差线的数据点。
        legend：添加图例。
        set_xlim 和 set_ylim：设置 x 轴和 y 轴的显示范围。
        set_xlabel 和 set_ylabel：设置 x 轴和 y 轴的标签。
        set_title：设置图表的标题。
        grid：显示或隐藏网格。
        set_facecolor 和 set_edgecolor：设置坐标轴的背景色和边缘色。
        set_alpha：设置坐标轴的透明度。
        set_axis_bgcolor 和 set_axis_color：设置坐标轴背景色和坐标轴颜色。
    '''
    ax1 = fig.add_subplot()
    x1 = feature1[0]
    y1 = feature1[1]
    # 画第一个图
    '''
    axes.plot 是 matplotlib.axes.Axes 类的一个方法，用于在坐标轴上绘制线条。它的参数可以用来控制线条的外观和行为。以下是一些常用的参数：
        x 和 y：这两个参数是必须的，它们分别代表线条的x轴和y轴坐标。x 可以是单个数值或数值数组，y 必须与 x 具有相同的形状。
        color：线条的颜色，可以是颜色名（如 ‘red’、‘green’ 等），十六进制颜色码（如 ‘#FF00FF’），RGB 或 RGBA 元组，或者预定义的颜色字符串（如 ‘C0’、‘C1’ 等，其中 ‘C0’、‘C1’ 等表示颜色循环中的颜色）。
        linewidth：线条的宽度，默认为 1.0。
        linestyle：线条的样式，可以是实线（‘-’）、虚线（‘–’）、点线（‘:’）、点（‘.’）等。
        marker：数据点的标记样式，如 ‘o’（圆形）、‘s’（方形）、‘^’（三角形上）、‘<’（三角形下）等。
        markersize：数据点的大小。
        markeredgewidth：数据点边缘线的宽度。
        markeredgecolor：数据点边缘线的颜色。
        label：用于图例的标签。
        alpha：线条和标记的透明度，范围从 0（完全透明）到 1（完全不透明）。
        ax：要绘制的坐标轴对象。
        data：如果 x 和 y 参数未提供，则可以提供一个包含 x 和 y 数据的字典或元组。
    '''
    # ax1.boxplot(x1, y1, 'r')
    x2 = feature2[0]
    print(x2)
    y2 = feature2[1]
    ax1.boxplot(x2, notch=False, patch_artist=True, labels=['A'])
    # 坐标轴微调
    '''
    在 matplotlib 中，Axes 对象有一个名为 axis 的方法，它用于设置和调整坐标轴的各种属性。这个方法通常接受一个参数，表示要调整的坐标轴（x 或 y），以及一些可选的参数来控制轴的显示和行为。
    以下是 axis 方法的一些常见用法：
        axis('off')：关闭坐标轴的显示。
        axis('on')：开启坐标轴的显示。
        axis('equal')：确保两个坐标轴的刻度比例相同，这对于绘制等比例的地图或其他图形非常重要。
        axis('scaled')：这是 axis('equal') 的一个别名。
        axis('tight')：自动调整坐标轴的范围以适应数据，通常在绘制多个子图时使用。
        axis('auto')：自动调整坐标轴的范围，通常是默认行为。
        axis([xmin, xmax, ymin, ymax])：手动设置坐标轴的范围。
        axis_bgcolor：设置坐标轴的背景色。
        axis_color：设置坐标轴的颜色。
    '''

    # 显示图表
    plt.show()
    # 将图表保存为文件。
    plt.savefig(filename, format='png', dpi=300)


if __name__ == '__main__':
    a = np.arange(1, 10)
    b = 2 * a
    c = 3 * a
    # 将a和b合并为一个二维数组
    feature1 = np.vstack((a, b))
    # feature2 = np.vstack((a, c))
    save_name = 'figure.png'
    data1 = [2, 2, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10]
    data2 = [1, 1, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 10, 9]
    feature2 = (data1, data2)
    plot_figure(feature1, feature2, save_name)
