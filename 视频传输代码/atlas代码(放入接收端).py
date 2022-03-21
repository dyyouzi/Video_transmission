import sys
sys.path.append("../../../../common")
sys.path.append("../")
import os
import numpy as np
import numpy
import acl
import cv2 as cv
from PIL import Image
import constants as const
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource
import time
import socket
import signal
from queue import Queue

labels =["person",  "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench","bird", 
        "cat", "dog", "horse", "sheep", "cow", 
        "elephant",  "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag",  "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball",  "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]
INPUT_DIR = '../data/'
OUTPUT_DIR = '../out/'
MODEL_PATH = "../model/yolov4_bs1.om"
MODEL_WIDTH = 608
MODEL_HEIGHT = 608
class_num = 80
stride_list = [32, 16, 8]
anchors_3 = np.array([[12, 16], [19, 36], [40, 28]]) / stride_list[2]
anchors_2 = np.array([[36, 75], [76, 55], [72, 146]]) / stride_list[1]
anchors_1 = np.array([[142, 110], [192, 243], [459, 401]]) / stride_list[0]
anchor_list = [anchors_1, anchors_2, anchors_3]
iou_threshold = 0.3
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


HOST = '192.168.1.8'
PORT = 9999
#PORT = 5555
buffSize = 50010

SEND_HOST = '192.168.1.176'
SEND_PORT = 8000


def preprocess(img_np):
# def preprocess(img_path):
    # image = Image.open(img_path)
    image = Image.fromarray(np.uint8(img_np))
    img_h = image.size[1]
    img_w = image.size[0]
    net_h = MODEL_HEIGHT
    net_w = MODEL_WIDTH

    scale = min(float(net_w) / float(img_w), float(net_h) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    shift_x = (net_w - new_w) // 2
    shift_y = (net_h - new_h) // 2
    shift_x_ratio = (net_w - new_w) / 2.0 / net_w
    shift_y_ratio = (net_h - new_h) / 2.0 / net_h

    image_ = image.resize((new_w, new_h))
    new_image = np.zeros((net_h, net_w, 3), np.uint8)
    new_image[shift_y: new_h + shift_y, shift_x: new_w + shift_x, :] = np.array(image_)
    new_image = new_image.astype(np.float32)
    new_image = new_image / 255
    print('new_image.shape', new_image.shape)
    new_image = new_image.transpose(2, 0, 1).copy()
    return new_image, image

def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(all_boxes, thres):
    res = []
    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]
        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue
            truth = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1
        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res

def _sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def decode_bbox(conv_output, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio):
    print('conv_output.shape', conv_output.shape)
    _, _, h, w = conv_output.shape 
    conv_output = conv_output.transpose(0, 2, 3, 1)
    pred = conv_output.reshape((h * w, 3, 5 + class_num))
    pred[..., 4:] = _sigmoid(pred[..., 4:])
    pred[..., 0] = (_sigmoid(pred[..., 0]) + np.tile(range(w), (3, h)).transpose((1, 0))) / w
    pred[..., 1] = (_sigmoid(pred[..., 1]) + np.tile(np.repeat(range(h), w), (3, 1)).transpose((1, 0))) / h
    pred[..., 2] = np.exp(pred[..., 2]) * anchors[:, 0:1].transpose((1, 0)) / w
    pred[..., 3] = np.exp(pred[..., 3]) * anchors[:, 1:2].transpose((1, 0)) / h

    bbox = np.zeros((h * w, 3, 4))
    bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, 0)  # x_min
    bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, 0)  # y_min
    bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, img_w)  # x_max
    bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, img_h)  # y_max
    pred[..., :4] = bbox
    pred = pred.reshape((-1, 5 + class_num))
    pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
    pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)    
    pred = pred[pred[:, 4] >= 0.2]
    print('pred[:, 5]', pred[:, 5])
    print('pred[:, 5] shape', pred[:, 5].shape)

    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4] - 1].append(box)
    return all_boxes

def convert_labels(label_list):
    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [labels[int(index)] for index in label_list]
    return label_names

def post_process(infer_output, origin_img):
    print("post process")
    result_return = dict()
    img_h = origin_img.size[1]
    img_w = origin_img.size[0]
    scale = min(float(MODEL_WIDTH) / float(img_w), float(MODEL_HEIGHT) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    shift_x_ratio = (MODEL_WIDTH - new_w) / 2.0 / MODEL_WIDTH
    shift_y_ratio = (MODEL_HEIGHT - new_h) / 2.0 / MODEL_HEIGHT
    class_number = len(labels)
    num_channel = 3 * (class_number + 5)
    x_scale = MODEL_WIDTH / float(new_w)
    y_scale = MODEL_HEIGHT / float(new_h)
    all_boxes = [[] for ix in range(class_number)]
    print(infer_output[0].shape)
    print(infer_output[1].shape)
    print(infer_output[2].shape)
    for ix in range(3):    
        pred = infer_output[ix]
        print('pred.shape', pred.shape)
        anchors = anchor_list[ix]
        boxes = decode_bbox(pred, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio)
        all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_number)]

    res = apply_nms(all_boxes, iou_threshold)
    if not res:
        result_return['detection_classes'] = []
        result_return['detection_boxes'] = []
        result_return['detection_scores'] = []
        return result_return
    else:
        new_res = np.array(res)
        picked_boxes = new_res[:, 0:4]
        picked_boxes = picked_boxes[:, [1, 0, 3, 2]]
        picked_classes = convert_labels(new_res[:, 4])
        picked_score = new_res[:, 5]
        result_return['detection_classes'] = picked_classes
        result_return['detection_boxes'] = picked_boxes.tolist()
        result_return['detection_scores'] = picked_score.tolist()
        return result_return


bgr_img = None
needExit = False
send_list = [[0,0,0,0,0,0,2]]

q_data = Queue(maxsize=100)

def recv3():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建接收对象
    server.bind((HOST, PORT))
    sum4=0
    while not needExit:
        data_1,addr=server.recvfrom(buffSize)
        q_data.put(data_1)
        if data_1[0] == 3:
            print("视频接收完毕！！")
            break;
    server.close()
    
    
def trsHandler():
    # 通信
    server_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # 创建发送对象
    #server_send.connect((SEND_HOST, SEND_PORT))
    print('sending {} data'.format("data_type"))
    #server_send.sendall('{}|{}'.format("rgb", "rgb").encode())
    print("trs")
    count = 0
    count1 = 0
    data_total=b''
    sum1 = 0
    sum2 = 0
    sum3 = 0
    first = np.uint8([1,255])
    q = Queue(maxsize = 10)
    
    while not needExit:
        # 图片
        #print("------waiting for image-----").
        while not q_data.empty():
            data = q_data.get()  # 接收编码图像数据
            #count = count + 1
            if first[0]==data[0]:
                first[1]=data[1]
                data_total+=data[2:]
                sum1+=1
            else:
                if sum1==first[1]:
                    nparr = np.frombuffer(data_total, np.uint8)
                    img_decode = cv.imdecode(nparr, cv.IMREAD_COLOR)
                    sum3+=1
                    q.put(img_decode)
                else:
                    sum2+=1
                    print("loss the frame!!")
                data_total=b''
                data_total+=data[2:]
    
                first[0]=data[0]
                first[1]=data[1]
                sum1 =1
            if data[0] == 3:
                print("OVER")
                break
        '''if count%3 == 0:
            data = numpy.array(bytearray(data))  # 格式转换
            imgdecode = cv.imdecode(data, 1)  # 解码
            que.put(imgdecode)'''
        if q.full():
            img = q.get()
            result, send_data = cv.imencode('.jpg', img, [cv.IMWRITE_JPEG_QUALITY, 20]) # 编码
            #print("send data len = " + str(len(send_data)))
            server_send.sendto('{}|{}|{}'.format('0', "data", "data").encode(),(SEND_HOST,SEND_PORT))
            server_send.sendto(send_data,(SEND_HOST,SEND_PORT))
            #server_send.sendall(np.array([[0,0,0,0,0,0,2]]).tobytes())
            server_send.sendto(np.array(send_list, dtype=np.int16).tobytes(),(SEND_HOST,SEND_PORT))
            
            global bgr_img
            bgr_img = img
    
    server_send.sendto('{}|{}|{}'.format('1', 0, "close").encode(),(SEND_HOST,SEND_PORT))
    server_send.close()

def ctrl_c(signalnum, frame):
    global needExit
    needExit = True

import threading
def main():
    signal.signal(signal.SIGINT, ctrl_c)
    signal.signal(signal.SIGTERM, ctrl_c)
    thread = threading.Thread(target=trsHandler)
    thread.start()
    
    thread1 = threading.Thread(target=recv3)
    thread1.start()
    
    '''
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    '''

    #ACL resource initialization
    acl_resource = AclLiteResource()
    acl_resource.init()
    #load model
    model = AclLiteModel(MODEL_PATH)
    images_list = [os.path.join(INPUT_DIR, img)
                   for img in os.listdir(INPUT_DIR)
                   if os.path.splitext(img)[1] in const.IMG_EXT]
    kill_start_time = time.time()
    #Read images from the data directory one by one for reasoning
    count = 0
    while not needExit:
        # bgr_img = cv.imread(pic)
        
        if bgr_img is not None:
            data, orig = preprocess(bgr_img)
            result_list = model.execute([data,])
            result_return = post_process(result_list, orig)
            print("result = ", result_return)
            send_res_list = []
            for i in range(len(result_return['detection_classes'])):
                class_name = result_return['detection_classes'][i]
                if class_name == 'car' or class_name == 'truck' or class_name == 'bus' or class_name == 'person':
                    res_list = [0, 0, 0, 0, 0, 1, 2]
                    if class_name == 'truck' or class_name == 'bus':
                        res_list[0] = 2
                    if class_name == 'person':
                        res_list[0] = 1
                    box = result_return['detection_boxes'][i]
                    confidence = result_return['detection_scores'][i]
                    res_list[1] = int(box[1])
                    res_list[2] = int(box[0])
                    res_list[3] = int(box[3])
                    res_list[4] = int(box[2])
                    #res_list[5] = format(confidence, '.2f')
                    send_res_list.append(res_list)
                    #cv.rectangle(bgr_img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), colors[i % 6])
                    #p3 = (max(int(box[1]), 15), max(int(box[0]), 15))
                    #out_label = class_name            
                    #cv.putText(bgr_img, out_label, p3, cv.FONT_ITALIC, 0.6, colors[i % 6], 1)
                    #print("orig_cls = " + str(class_name))
                    #print("change_cls = " + str(res_list[0]))
        #output_file = os.path.join(OUTPUT_DIR, "out_" + str(count) + ".jpg")
        #print("output:%s" % output_file)
        #cv.imwrite(output_file, bgr_img)

            # 传输
            print("send res list")
            global send_list
            send_list = send_res_list
        kill_end_time = time.time()
        if kill_end_time - kill_start_time > 300:
            break
    
    thread1.join()
    thread.join()
    print("Execute end")
    exit(0)

if __name__ == '__main__':
    main()
 
