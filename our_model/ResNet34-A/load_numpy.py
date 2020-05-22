import os, glob
import random, csv
import cv2

from PIL import Image
import numpy as np
from numpy import *
import os
'''数据集路径'''
data_path='/CZC/restnet_junk_34/junk_test1/dataset_resize'


def load_csv(root, filename, name2label):
    """
    加载CSV文件！
    :param root:
    :param filename:
    :param name2label:
    :return:
    """
    # root:数据集根目录
    # filename:csv文件名
    # name2label:类别名编码表
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
            # 'pokemon\\mewtwo\\00001.png
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))

        # 1167, 'pokemon\\bulbasaur\\00000000.png'
        print(len(images), images)

        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                name = img.split(os.sep)[-2]
                label = name2label[name]
                # 'pokemon\\bulbasaur\\00000000.png', 0
                writer.writerow([img, label])
            print('written into csv file:', filename)

    # read from csv file
    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            # 'pokemon\\bulbasaur\\00000000.png', 0
            img, label = row
            label = int(label)

            images.append(img)
            labels.append(label)

    assert len(images) == len(labels)

    return images, labels


# 加载pokemon数据集的工具！
def load_pokemon(root, mode='train'):
    """
    加载pokemon数据集的工具！
    :param root:    数据集存储的目录
    :param mode:    mode:当前加载的数据是train,val,还是test
    :return:
    """
    # 创建数字编码表
    name2label = {}  # "sq...":0   类别名:类标签;  字典 可以看一下目录,一共有6个文件夹,6个类别：0-5范围;
    for name in sorted(os.listdir(os.path.join(root))):  # 列出所有目录;
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())

    # 读取Label信息;保存索引文件images.csv
    # [file1,file2,], 对应的标签[3,1] 2个一一对应的list对象。
    images, labels = load_csv(root, 'images.csv', name2label)  # 根据目录,把每个照片的路径提取出来,以及每个照片路径所对应的类别都存储起来，存储到CSV文件中。
    if mode == 'train':  # 60%
        images = images[:int(0.8 * len(images))]
        labels = labels[:int(0.8 * len(labels))]
    elif mode == 'test':  # 20% = 60%->80%
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]
    return images, labels, name2label
def getdata(data=data_path,model=''):
    images, labels, table = load_pokemon(data, model)
    labels=np.array(labels)
    i=0
    sum_rgb = []
    sum_img = []
    count=0
    for path in images:
        img = Image.open(path, 'r')
        r, g, b = img.split()
        r=np.array(r)
        r=cv2.resize(r,(64,64),interpolation=cv2.INTER_CUBIC)
        g = np.array(g)
        g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_CUBIC)
        b= np.array(b)
        b = cv2.resize(b, (64, 64), interpolation=cv2.INTER_CUBIC)
        # print (np.array(r).shape)
        sum_rgb.append(b)
        sum_rgb.append(g)
        sum_rgb.append(r)
        # print (np.array(sum_rgb).shape)
        sum_img.append(sum_rgb)
        # print (np.array(sum_img).shape)
        sum_rgb = []
        count = count + 1
    sum_img=np.array(sum_img)
    sum_img=sum_img.swapaxes(1,3)
    sum_img=sum_img.swapaxes(2,1)
    return sum_img,labels

# x_test,y_test=getdata(model='test')
# x_train,y_train=getdata(model='train')
# x_train=np.reshape(x_train,[224,224,3])
# cv2.imshow('111',x_train)
# cv2.waitKey(0)