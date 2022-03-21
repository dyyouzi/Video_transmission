# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:30:54 2022

@author: Admin
"""

#用两个线程，一个收，一个展示
#对帧进行排序，掉了帧，下一个帧到来了前一个帧要丢弃
import socket
import cv2
import numpy as np
import struct
import threading
import time
from queue import Queue
import signal

HOST='127.0.0.1'
PORT=9999
q = Queue(maxsize=0) #帧存放队列
q_data = Queue(maxsize=30)
size = 30000+2
needExit = False
show_c = True #用于控制show函数顺利退出，发送端发来结束包，则变为false

def ctrl_c(signalnum, frame):
    global needExit
    needExit = True

def recv3():

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 绑定端口:
    s.bind((HOST, PORT))
    print('Bind UDP on 9999...')
    sum4 = 0 #统计数据包个数
    while not needExit:
        data_1,addr = s.recvfrom(size)
        q_data.put(data_1)
        sum4+=1
        print("q_data=",q_data.qsize())
        #接收到包头为3的数据包则结束
        if data_1[0]==3:
            print("数据包接收完毕")
            break
     
def trsHandler():
    data_total = b''
    sum1 = 0                        #统计当前帧一共收到多少数据包
    sum2 = 0                        #统计视频帧丢了多少
    sum3 = 0                        #统计完整视频帧接收了多少
    first = np.uint8([1,255])         #存放前面数据包的数据头，b初始化要为1不能为0,使得接收方随时可以开启接受，
					       #否则当接受第一个数据包时，若初始化a和data[0]不相等时
					#初始化b和sum1相等，从而使程序报错
    while not needExit:
        #接收数据包
        where not q_data.empty() :
            data = q_data.get()
        #前一个数据包和当前数据包相同,说明属于同一帧
            if first[0] == data[0]:
                first[1] = data[1]  #获取数据分片数量
                data_total += data[2:]
                sum1 += 1  
            #前一个数据包和当前数据包不相同，说明属于不同帧
            else:
                if sum1 == first[1]:            #获取的数据包没有遗漏，帧完整放入队列
                    nparr = np.frombuffer(data_total, np.uint8) #还原数组
                    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR) #将数组变成图片
                    sum3+=1                     #完整帧的个数
                    q.put(img_decode)
                else:                           #获取的数据包有遗漏，帧不完整不放入队列
                    sum2+=1                     #不完整帧的个数
                    #print("该帧丢失！！！")
                #重新接收新的帧
                data_total=b'' #重新获取数据，之前数据清空
                data_total += data[2:]
                
                first[0]=data[0] #获取该帧的数据头
                first[1]=data[1]
                
                sum1=1 #数据包数量重新计数
            
            if data[0]==3:
                print("视频帧接收完毕")
                print("总共收到的完整帧为",sum3)
                print("丢失帧的个数为",sum2)
                global show_c
                show_c=False  
                break
        
def show():
    print("图片展示")
    while not needExit:
        if not q.empty():
            img_decode = q.get()
            cv2.namedWindow("result",0);
            cv2.resizeWindow("result", 640, 480);
            cv2.imshow('result',img_decode)
            cv2.waitKey(10)
            time.sleep(0.01)  #60fps
        else:
            if not show_c:
                break
            
def main():
    signal.signal(signal.SIGINT, ctrl_c)
    signal.signal(signal.SIGTERM, ctrl_c)
    thread = threading.Thread(target=trsHandler)
    thread.start()
    
    thread1 = threading.Thread(target=show)
    thread1.start()
    
    thread2 = threading.Thread(target=recv3)
    thread2.start()

    
    thread.join()
    thread1.join()
    thread2.join()
    print("Execute end")
    #exit(0)
    
if __name__ == '__main__':
    main()


