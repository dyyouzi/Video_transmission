

import cv2
import numpy as np
import socket
import struct
import time
HOST='127.0.0.1'
PORT=9999
size = 30000
address = 'video.mp4'

s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) #socket对象
print('now starting to send frames...')
capture=cv2.VideoCapture(address) #VideoCapture对象，可获取摄像头设备的数据
if not capture.isOpened():
   print("Error opening video!!!!")
A =10 
a = 1  
sum2=0 #统计一共发了多少帧
sum1=0 #统计一共发了多少数据包
while A:
    #A -=1
    success,frame=capture.read() #获取视频帧
    if not success:
       print("Camera read error!!!!")
       #视频发送完毕，发送结束数据包，包头为3表示发送结束
       data = np.zeros((size+2,), dtype=np.int8()) #创建一个大小为size+2内容空的数组
       data[0] = np.int8(3)                         #改变包头序号
       s.sendto(data,(HOST,PORT))
       print("视频发送完毕,共发送",sum2,"帧")
       print("视频发送完毕,共发送",sum1,"数据包")
       break
    
    #对视频帧进行编码
    result,img_encode=cv2.imencode('.jpg',frame) 
    data_encode = np.array(img_encode)

    
    b = len(data_encode)//size+1
    print("发送的帧序号=",a," 要发送的数据包总数=",b)
    for i in range(b):
        first = np.int8([a,b])  #创建数据头

        if size*(i+1)>len(data_encode):
            d1 = data_encode[size*i:] #获取数据
            d2 = np.insert(d1,0,first) #加入数据头
            data = d2.tobytes()   #将数据变为字节流
            s.sendto(data, (HOST, PORT))
            sum1+=1
        else:
            d1 = data_encode[size*i:size*(i+1)] #获取数据
            d2 = np.insert(d1,0,first) #加入数据头
            data = d2.tobytes()   #将数据变为字节流#if()
            s.sendto(data,(HOST,PORT))
            sum1+=1
    sum2+=1
    if a==1:
        a=2
    else:
        a=1
    print('have sent one frame')



    
