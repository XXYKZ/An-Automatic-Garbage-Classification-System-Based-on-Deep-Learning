import json
import matplotlib.pyplot as plt
val_acc='/CZC/garbage//val_acc.txt'
loss='/CZC/garbage/标准resnet34/loss.txt'
def read_txt(path=''):
    f = open(path)
    list = f.read()
    list = json.loads(list)
    return list
acc=read_txt(val_acc)
loss=read_txt(loss)
print(acc[99])
print(loss[99])
