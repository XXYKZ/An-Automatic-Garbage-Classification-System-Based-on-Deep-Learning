# -*- coding: utf-8 -*-
#!/usr/bin/env python    

import cv2
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO  
import time  
import signal  
import atexit
import datetime

atexit.register(GPIO.cleanup)
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)
GPIO.output(26, True)
time.sleep(4)
GPIO.output(26, False)
GPIO.cleanup()

start=datetime.datetime.now()


servopin = 19   #number 32   down
servopin1 = 13   #number 12  top
GPIO.setmode(GPIO.BCM)
GPIO.setup(servopin, GPIO.OUT, initial=False)
GPIO.setup(servopin1, GPIO.OUT, initial=False)
p = GPIO.PWM(servopin,50) #50HZ
p1 = GPIO.PWM(servopin1,50) #50HZ 
p.start(0)
p1.start(0)

p.ChangeDutyCycle(2.5)
time.sleep(0.02)                      #等该20ms周期结束  
p.ChangeDutyCycle(0)
time.sleep(0.18)

p1.ChangeDutyCycle(2.5)
time.sleep(0.02)                      #等该20ms周期结束  
p1.ChangeDutyCycle(0)
time.sleep(0.18)

time.sleep(2)


# when it can not be resolve
for a in range(1):
   
    for i in range(0,91,90):  
      p1.ChangeDutyCycle(2.5 + 5.8* i / 180) #设置转动角度  
      time.sleep(0.02)                      #等该20ms周期结束  
      p1.ChangeDutyCycle(0)                  #归零信号  
      time.sleep(0.18)
      
        
    for i in range(0,91,90):  
      p.ChangeDutyCycle(2.5 + 5.8* i / 180) #设置转动角度  
      time.sleep(0.02)                      #等该20ms周期结束  
      p.ChangeDutyCycle(0)                  #归零信号  
      time.sleep(0.18)
      time.sleep(0.2)
         
      
    for i in range(91,0,-90):  
      p.ChangeDutyCycle(2.5 + 5.8* i / 180) #设置转动角度  
      time.sleep(0.02)                      #等该20ms周期结束  
      p.ChangeDutyCycle(0)                  #归零信号  
      time.sleep(0.18)
      
      
    for i in range(91,0,-90):  
      p1.ChangeDutyCycle(2.5 + 5.8* i / 180) #设置转动角度  
      time.sleep(0.02)                      #等该20ms周期结束  
      p1.ChangeDutyCycle(0)                  #归零信号  
      time.sleep(0.18)
end=datetime.datetime.now()
print(end-start)
time.sleep(3)



