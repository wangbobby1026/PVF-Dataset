"""
@FileName：Matrix.py\n
@Description：混淆矩阵\n
@Author：WBobby\n
@Department：CUG\n
@Time：2023/7/12 10:05\n
"""
import csv

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def read_csv(infile):
    df = pd.read_csv(infile)
    data = df.values.tolist()
    for row in data:
        image = row[1]
        row.append(image[:2])
    return data

def matrix(data,outfile):
    y_true = [row[2] for row in data]
    # print(y_true)
    y_pred = [row[4] for row in data]
    # print(y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred)
    # print(confusion_mat)
    return confusion_mat









if __name__ == '__main__':
    infile = r'C:\Users\Wbobby\Desktop\BB.csv'
    outfile = ''
    data = read_csv(infile)
    matrix(data, outfile)