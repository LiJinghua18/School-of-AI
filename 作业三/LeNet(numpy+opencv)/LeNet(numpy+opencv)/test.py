# -*- coding: utf-8 -*-
import numpy as np
import fetch_MNIST
import cv2


class LeNet(object):
    def __init__(self, lr=0.1):
        self.lr = lr
        self.conv1 = xavier_init(6, 1, 5, 5)
        self.pool1 = [2, 2]
        self.conv2 = xavier_init(16, 6, 5, 5)
        self.pool2 = [2, 2]
        self.fc1 = xavier_init(256, 200, fc=True)
        self.fc2 = xavier_init(200, 10, fc=True)

    def forward_prop(self, input_data):
        self.l0 = np.expand_dims(input_data, axis=1) / 255
        self.l1 = self.convolution(
            self.l0, self.conv1) 
        self.l2 = self.mean_pool(self.l1, self.pool1) 
        self.l3 = self.convolution(self.l2, self.conv2) 
        self.l4 = self.mean_pool(self.l3, self.pool2)  
        self.l5 = self.fully_connect(self.l4, self.fc1)  
        self.l6 = self.relu(self.l5)  
        self.l7 = self.fully_connect(self.l6, self.fc2)  
        self.l8 = self.relu(self.l7) 
        self.l9 = self.softmax(self.l8)  
        return self.l9

    def backward_prop(self, softmax_output, output_label):
        l8_delta = (output_label - softmax_output) / softmax_output.shape[0]
        l7_delta = self.relu(self.l8, l8_delta, deriv=True)  
        l6_delta, self.fc2 = self.fully_connect(
            self.l6, self.fc2, l7_delta, deriv=True)  
        l5_delta = self.relu(self.l6, l6_delta, deriv=True)  
        l4_delta, self.fc1 = self.fully_connect(
            self.l4, self.fc1, l5_delta, deriv=True)  
        l3_delta = self.mean_pool(
            self.l3, self.pool2, l4_delta, deriv=True)  
        l2_delta, self.conv2 = self.convolution(
            self.l2, self.conv2, l3_delta, deriv=True)  
        l1_delta = self.mean_pool(self.l1, self.pool1, l2_delta, deriv=True)
        l0_delta, self.conv1 = self.convolution(
            self.l0, self.conv1, l1_delta, deriv=True)  

    def convolution(self, input_map, kernal, front_delta=None, deriv=False):
        N, C, W, H = input_map.shape
        K_NUM, K_C, K_W, K_H = kernal.shape
        if deriv == False:
            feature_map = np.zeros((N, K_NUM, W - K_W + 1, H - K_H + 1))
            for imgId in range(N):
                for kId in range(K_NUM):
                    for cId in range(C):
                        feature_map[imgId][kId] += \
                            conv_simple(
                                input_map[imgId][cId], kernal[kId, cId, :, :])
            return feature_map
        else:
            back_delta = np.zeros((N, C, W, H))
            kernal_gradient = np.zeros((K_NUM, K_C, K_W, K_H))
            padded_front_delta = \
                np.pad(front_delta, [(0, 0), (0, 0), (K_W - 1, K_H - 1), (K_W - 1, K_H - 1)], mode='constant',
                       constant_values=0)
            for imgId in range(N):
                for cId in range(C):
                    for kId in range(K_NUM):
                        back_delta[imgId][cId] += \
                            conv_simple(
                                padded_front_delta[imgId][kId], kernal[kId, cId, ::-1, ::-1])
                        kernal_gradient[kId][cId] += \
                            conv_simple(
                                front_delta[imgId][kId], input_map[imgId, cId, ::-1, ::-1])
            kernal += self.lr * kernal_gradient
            return back_delta, kernal

    def mean_pool(self, input_map, pool, front_delta=None, deriv=False):  # 简化：没有预留padding操作
        N, C, W, H = input_map.shape
        P_W, P_H = tuple(pool)
        out_W, out_H = W // P_W, H // P_H
        if deriv == False:
            feature_map = np.zeros((N, C, out_W, out_H))
            for n in np.arange(N):
                for c in np.arange(C):
                    for i in np.arange(out_W):
                        for j in np.arange(out_H):
                            feature_map[n, c, i, j] = np.mean(
                                input_map[n, c, P_W*i:P_W*(i+1), P_H*j:P_H*(j+1)])
            return feature_map
        else:
            back_delta = np.zeros((N, C, W, H))
            back_delta = front_delta.repeat(P_W, axis=2).repeat(P_H, axis=3)
            back_delta /= (P_W * P_H)
            return back_delta

    def fully_connect(self, input_data, fc, front_delta=None, deriv=False):
        N = input_data.shape[0]
        if deriv == False:
            output_data = np.dot(input_data.reshape(N, -1), fc)
            return output_data
        else:
            back_delta = np.dot(front_delta, fc.T).reshape(input_data.shape)
            fc += self.lr * np.dot(input_data.reshape(N, -1).T, front_delta)
            return back_delta, fc

    def relu(self, x, front_delta=None, deriv=False):
        if deriv == False:
            return x * (x > 0)
        else:
            back_delta = front_delta * 1. * (x > 0)
            return back_delta

    def softmax(self, x):
        y = list()
        for t in x:
            e_t = np.exp(t - np.max(t))
            y.append(e_t / e_t.sum())
        return np.array(y)


def xavier_init(c1, c2, w=1, h=1, fc=False):
    fan_1 = c2 * w * h
    fan_2 = c1 * w * h
    ratio = np.sqrt(6.0 / (fan_1 + fan_2))
    params = ratio * (2 * np.random.random((c1, c2, w, h)) - 1)
    if fc == True:
        params = params.reshape(c1, c2)
    return params


def conv_simple(img, kernal, stride=1):
    # 检查feature与kernel的大小关系，否则调换
    ok1, ok2 = True, True
    shape1 = img.shape
    shape2 = kernal.shape
    if not shape1[0] >= shape2[0]:
        ok1 = False
    if not shape1[1] >= shape2[1]:
        ok2 = False
    if ok1 == False and ok2 == False:
        temp = img
        img = kernal
        kernal = temp
    elif (ok1 == True and ok2 == False) or (ok2 == True and ok1 == False):
        raise ValueError('请检查img和kernal的shape')

    W, H = img.shape
    K_W, K_H = kernal.shape
    out_w = (W-K_W)//stride+1
    out_h = (H-K_H)//stride+1
    feature_map = np.zeros((out_w, out_h))
    for i in range(0, out_w-1, stride):
        for j in range(0, out_h-1, stride):
            feature_map[i][j] = np.sum(np.multiply(
                img[i:i+K_W, j:j+K_H], kernal))
    return feature_map


def save_para(net, path):
    np.savez(path, conv1=net.conv1, pool1=net.pool1,
             conv2=net.conv2, pool2=net.pool2, fc1=net.fc1, fc2=net.fc2)


def load_para(net, path):
    para = np.load(path)
    net.conv1 = para['conv1']
    net.pool1 = para['pool1']
    net.conv2 = para['conv2']
    net.pool2 = para['pool2']
    net.fc1 = para['fc1']
    net.fc2 = para['fc2']


def convertToOneHot(labels):
    oneHotLabels = np.zeros((labels.size, labels.max() + 1))
    oneHotLabels[np.arange(labels.size), labels] = 1
    return oneHotLabels


def shuffle_dataset(data, label):
    N = data.shape[0]
    index = np.random.permutation(N)
    x = data[index, :, :]
    y = label[index, :]
    return x, y


def mouse_event(event, x, y, flags, param):
    global start, drawing, mode

    # 左键按下：开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)
    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 12, (255, 255, 255), -1)
    # 左键释放：结束画图
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 12, (255, 255, 255), -1)


if __name__ == '__main__':
    lr = 0.01
    path = '5508.npz'
    my_CNN = LeNet(lr)
    if path is not None:
        load_para(my_CNN, path)

    drawing = False  # 是否开始画图
    start = (-1, -1)

    img = np.full((280, 280, 3), 0, np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_event)
    while(True):
        cv2.imshow('image', img)
        if cv2.waitKey(1) == 27:
            break
        elif cv2.waitKey(1) == ord('s'):
            cv2.imwrite('img.jpg', img)

            # img='img.jpg'
            # img=cv2.imread(img)
            img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imshow('image', img1)
            img1 = cv2.resize(img1, (28, 28))
            img1 = img1[np.newaxis, :]
            softmax_output = my_CNN.forward_prop(img1)
            print('*'*5)
            print('*',list(softmax_output[0]).index(max(list(softmax_output[0]))),'*')
            print('*'*5)
        elif cv2.waitKey(1) == ord('n'):
            img = np.full((280, 280, 3), 0, np.uint8)
