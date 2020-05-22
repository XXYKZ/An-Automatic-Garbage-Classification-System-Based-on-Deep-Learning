// init_gpio.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include "jetsonGPIO.h"
using namespace std;

// TX2上电，默认输出3.3V，连接STM32时测为1.5V，故开机时下面的舵机会转动一次
// TX2上电，默认输出0V，连接STM32时测为0V，故开机时下面的舵机不会转动
int main(int argc, char *argv[]){

    jetsonTX2GPIONumber TOP = gpio254 ;
    gpioExport(TOP) ;
    gpioSetDirection(TOP,outputPin) ;
    // 必须进行置0操作，否则此引脚会在其他引脚工作前保持高电平输出，这样会致使gpio255等引脚不正常工作
    gpioSetValue(TOP, off);

    jetsonTX2GPIONumber LOW = gpio255 ;     
    gpioExport(LOW) ;
    gpioSetDirection(LOW,outputPin) ;
    // 此引脚默认0V输出，故不必置0
    return 0;
}


