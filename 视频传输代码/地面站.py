# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 13:55:28 2022

@author: Admin
"""

import cv2
import socket
import numpy as np
import struct
import threading
import time
from queue import Queue
import signal

HOST = '127.0.0.1'
PORT = 5555
buffsize = 60000
Control = 0;
s1 = "data"
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

def ctrl_c(signalnum, frame):
    global needExit
    needExit = True


def server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    
    while not Control:
        #接收数据头，提示要接收数据了
        print("recv Fisrt!!")
        First,addr1 = sock.recvfrom(buffsize)
        str_First = First.decode().split('|')
        
        if str_First[1] is s1:
            print("视频接收完毕！！")
            break
        
        #接收数据帧信息
        print("recv second!!")
        second,addr1 = sock.recvfrom(buffsize)
        frame = np.array(bytearray(second))
        img_decode = cv2.imdecode(frame, 1)
        
        #接收坐标和种类信息
        print("recv third!!")
        third,addr1 = sock.recvfrom(buffsize)
        res_list = np.frombuffer(third, np.uint16)
        
        #将坐标信息在帧图片上画上框
        i=0 
        name=""
        #计算有几组坐标
        counts = len(res_list)//7
        print(len(res_list),",",counts)
        #将每组坐标分别加入框中
        for k in range(counts):
            i+=1
            cv2.rectangle(img_decode, (res_list[k*7+1], res_list[k*7+2]), (res_list[k*7+3], res_list[k*7+4]), colors[i % 6])
            p3 = (max(res_list[k*7+1], 15), max(res_list[k*7+2], 15))
            if res_list[k*7]==2:
                name = "truck"
            elif res_list[k*7]==1:
                name = "person"
            else:
                name = "car"
            cv2.putText(img_decode, name, p3, cv2.FONT_ITALIC, 0.6, colors[i % 6], 1)
        
        cv2.imshow('result',img_decode)
        cv2.waitKey(10)

def main():
    signal.signal(signal.SIGINT, ctrl_c)
    signal.signal(signal.SIGTERM, ctrl_c)
    server()
    
    print("Execute end")
    exit(0)

if __name__ == '__main__':
    main()
            
            
            
        
        
        
        
