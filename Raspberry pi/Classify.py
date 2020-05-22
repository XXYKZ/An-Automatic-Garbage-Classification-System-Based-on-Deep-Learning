import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import time
import cv2
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO  
import time  
import signal  
import atexit

atexit.register(GPIO.cleanup)
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)
GPIO.output(26, True)
time.sleep(1)
GPIO.output(26, False)
GPIO.cleanup()


servopin = 19   # 下舵机
servopin1 = 13  # 上舵机
GPIO.setmode(GPIO.BCM)
GPIO.setup(servopin, GPIO.OUT, initial=False)
GPIO.setup(servopin1, GPIO.OUT, initial=False)
p = GPIO.PWM(servopin,50) #50HZ
p1 = GPIO.PWM(servopin1,50) #50HZ 
p.start(0)
p1.start(0)

p.ChangeDutyCycle(2.1 + 7*40/180)
time.sleep(0.02)                      #等该20ms周期结束  
p.ChangeDutyCycle(0)
time.sleep(0.18)

p1.ChangeDutyCycle(1.5 + 7*90/180)
time.sleep(0.02)                      #等该20ms周期结束  
p1.ChangeDutyCycle(0)
time.sleep(0.18)

time.sleep(1)




# import servo
# 禁用TensorFlow编译警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 参考retrained_labels.txt
# book
# orange
# towel(0.20954)
# leaf
# plastic(0.2)
# bottle(0.27658)
# paperstrip(0.23979)
# papergroup(0.36710)
# banana
# paperconfetti(0.50167)
# 测试过程中发生异常不断修改阈值数据，或大或小
threshold = [0.33075,0.32589,0.26954,0.5,0.26,0.37,0.23979,0.37710,0.2,0.52667]   
# 错误的想法：仅设置分数排行榜可达到第一名物体的阈值
# 调用控制台参数
# image_path = sys.argv[1]
# 分析数据时，绘图使用
# object_score = [[] for i in range(8)]
# object_div = [[] for i in range(8)]
# object_var = []
# object_time = []

#
if __name__=='__main__':
    time_start=time.time()

blank_sorce = []
# OpenCv 调用摄像头
def get_image_data():
    # 默认摄像头为0,其他摄像头调用0  (1)
    cap = cv2.VideoCapture(0)
#    time.sieep(0.5)
    # cap.get(3)
    # 不必设置
    # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = 640  # 640
    height = 480 # 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    #  print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 判断视频读取或者摄像头调用是否成功，成功则返回true。
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            #cv2.imshow("capture", frame)
            cap.release()
            cv2.imwrite('tmp.jpg', frame)  # 存储为图像

            # image_data1 = frame.tobytes() # 非指定格式图片
            # Expected image (JPEG, PNG, or GIF), got unknown format starting with

            image_path = 'tmp.jpg'
            image_data = tf.gfile.GFile(image_path, 'rb').read()
            return image_data
        else:
            return 0
    return 0


# 从文件中获取标签
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("/home/pi/LGL/trabbish/tf_files/retrained_labels.txt")]

# 打开retrained_graph.pb文件进行二进制读取。
with tf.gfile.GFile("/home/pi/LGL/trabbish/tf_files/retrained_graph.pb", 'rb') as f:
    # graph-def是TensorFlow图的保存副本。
    # 首先我们需要创建一个空的graph-def
    graph_def = tf.GraphDef()
    # 然后我们将proto-buf文件加载到graph-def中
    graph_def.ParseFromString(f.read())  # 将序列化协议缓冲区数据解析为变量
    # 最后我们将graph-def导入默认TensorFlow图中
    _ = tf.import_graph_def(graph_def, name='')  # 导入序列化的TensorFlow GraphDef协议缓冲区，将GraphDef中的对象提取为tf.Tensor

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    # return: Tensor("final_result:0", shape=(?, 8), dtype=float32); stringname definiert in retrain.py, at 1064 line
    #tmp_flag = True
    # ii = 0
    #os.system('./init_gpio')
    #threshold = 0.6 
    while True:
        # if ii == 5:
        #     break
        # ii = ii + 1
        # print(ii)
        # start = time.time()
        image_data = get_image_data()
        if image_data != 0:
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            # 返回预测值数组
            # 0，1，2，3，4，5，6，7
            ## 无效：['bumf', 'banana', 'book', 'bottle', 'towel', 'plastic', 'orange', 'caiye']
        # 再次训练发生变更
        # 参考retrained_labels.txt
        # book
        # orange
        # towel
        # strip
        # plastic
        # bottle
        # banana
        # shuye
        # group
            # 其中不可回收索引为1，6,7
            # https://github.com/koflerm/tensorflow-image-classifier


            #if tmp_flag:
                #tmp_flag = False
                #tmp = predictions[0]
                #tmp_var = np.var(predictions[0])
            #else:
                #tmp_div = np.true_divide(predictions[0], tmp)
                # 记录物体差值
                # for i in range(0, 8):
                #     object_div[i].append(tmp_div[i])

                # top_k = tmp_div.argsort()[-len(tmp_div):][::-1]

                # 取得最大值索引
                # div_max_index = np.argmax(tmp_div)

                # for node_id in top_k:
                #     # human_string = label_lines[node_id]
                #     value = tmp_div[node_id]
                #     # print('tmp_div : %s (value = %.5f)' % (human_string, value))
                # 得分变化率
                #div_value = tmp_div[div_max_index]
                #tmp = predictions[0]

                # 方差变化率
                #var_value = np.var(predictions[0])/tmp_var
                #tmp_var = np.var(predictions[0])

                # print('\n')
                # print(label_lines[div_max_index], div_value)
                # print('\n')
                # print(arr_var,var_value)
                # 灵敏度 ：
                # lmd = div_value * var_value
                # print(lmd)
                # 取得新的最大值索引
            max_index = np.argmax(predictions[0])
            gate_score = predictions[0][max_index]
        # debug
            #print('\n')
            #top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            #for node_id in top_k:
               #human_string = label_lines[node_id]
               #score = predictions[0][node_id]
               #print('%s (score = %.5f)' % (human_string, score))
               #time_end=time.time()
               #print('usetime:',time_end-time_start)
               #break

                # 灵敏度
                #lmd = (gate_score > 0.6 or (gate_score > 0.5 and div_value > 2.2) or (gate_score > 0.4 and div_value > 2.6) or (gate_score > 0.3 and div_value > 4.8) or div_value > 6) and var_value > 1.0
                #if max_index == 0 :
                #    threshold =1.1 
                #elif max_index == 6:
                #    threshold = 0.4
                #else :
                #    threshold = 0.6
                # lmd1 = (gate_score + div_value/10.0 > threshold) and (gate_score > 0.2)
            lmd1 = (gate_score > threshold[max_index])
            if lmd1:
                # if  div_value > 2.6 or (div_value >2.15 and lmd > 0.0354 ):
                    # 考虑到初始环境各个物体分数差距较大,更，但是识别可不可回收足矣，此行可注释
                    # predictions[0][div_max_index] = predictions[0][div_max_index] + div_value/10.0
                #top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                #for node_id in top_k:
                    #human_string = label_lines[node_id]
                    #score = predictions[0][node_id]
                    #print('%s (score = %.5f)' % (human_string, score))
                    # 取得新的最大值索引
                max_index = np.argmax(predictions[0])
                if max_index == 1 or max_index == 3 or max_index == 8:
                        # servo.operate_servo(top=True)
                        # os.system可以做到线程阻塞，这个模块是用C实现的
                        # 先是fork了一个子线程，然后父线程会waitpid,缺点是返回值并不是真是的值（但0是0）
                    print('%s:Non-recyclable' % label_lines[max_index])
                    print(gate_score)
                    for i in range(0,181,45):  
                      p.ChangeDutyCycle(2.1 + 7 *(90 + i) / 180) #设置转动角度  
                      time.sleep(0.02)                      #等该20ms周期结束  
                      p.ChangeDutyCycle(0)                  #归零信号  
                      time.sleep(0.18)  

                    for i in range(181,0,-45):  
                      p.ChangeDutyCycle(2.1 + 7 *(40+ i) / 180)  
                      time.sleep(0.02)  
                      p.ChangeDutyCycle(0)  
                      time.sleep(0.18)  
                    #os.system('sudo ./low')
                        # 摄像头睡眠时间，舵机操作时间
                    time.sleep(0.5) # 二分法取值
                        # break
                else:
                        # operate_servo(top=True)
                    print('%s:Recyclable可回收' % label_lines[max_index]) 
                    # 上舵机
                    print(gate_score)
                    for i in range(0,226,45):  
                      p1.ChangeDutyCycle(1.6 + 7 * (90+i) / 180) #设置转动角度  
                      time.sleep(0.02)                      #等该20ms周期结束  
                      p1.ChangeDutyCycle(0)                  #归零信号  
                      time.sleep(0.18)  

                    for i in range(226,0,-45):  
                      p1.ChangeDutyCycle(1.6 + 7 *(90+i) / 180)  
                      time.sleep(0.02)  
                      p1.ChangeDutyCycle(0)  
                      time.sleep(0.18)  
                    #os.system('sudo ./top')
                        # 摄像头睡眠时间，舵机操作时间
                    time.sleep(0.5)
                        #
                        # break
            # 记录识别时间
            # object_time.append(time.time() - start)
            # print('时间:', time.time() - start)
            # top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            # 概率值从大到小的排列序号

            # 记录物体分值
            # for i in range(0, 8):
            #     object_score[i].append(predictions[0][i])
            # output
            # flag = True
            # wait = 0
            # for node_id in top_k:
            #     human_string = label_lines[node_id]
            #     score = predictions[0][node_id]
            #     print('%s (score = %.5f)' % (human_string, score))

            # 记录方差
            # object_var.append(arr_var)
            # print('方差:', arr_var)
        else:
            print("未采集到图像数据,系未接入摄像头")
            break


# def transpose(matrix):
#     new_matrix = []
#     for i in range(len(matrix[0])):
#         matrix1 = []
#         for j in range(len(matrix)):
#             matrix1.append(matrix[j][i])
#         new_matrix.append(matrix1)
#     return new_matrix


# plt.plot(object_time)
# plt.show()
# plt.plot(object_var)
# plt.show()
#
# plt.plot(transpose(object_score))
# plt.show()
# plt.plot(transpose(object_div))
# plt.show()
# print('')
#if flag:
#        flag = False
     # 用用于显示
    #image = cv2.imread(image_path)
 # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    #cv2.putText(image, human_string + ':' + '%.5f' % score, (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    #cv2.imshow('Image', image)
    #cv2.waitKey(wait)
