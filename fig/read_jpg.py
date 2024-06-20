"""
@FileName：read_jpg.py\n
@Description：\n
@Author：WBobby\n
@Department：CUG\n
@Time：2024/3/26 23:07\n
"""
import cv2

if __name__ == '__main__':
    jpg = r'C:\Users\Wbobby\Desktop\12\DJI_20231225162517_0002_T.JPG'
    img = cv2.imread(jpg)
    cv2.imshow('image', img)
    cv2.waitKey(0)
