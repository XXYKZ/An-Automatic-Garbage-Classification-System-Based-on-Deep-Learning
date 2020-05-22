// top.c

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

int main(int argc, char *argv[]){

    //cout << "active TOP" << endl;
    jetsonTX2GPIONumber TOP = gpio254 ;     // Ouput
    // Make the TOP available in user space
    //gpioExport(TOP) ;
    //gpioSetDirection(TOP,outputPin) ;
    //cout << "Setting the TOP off" << endl;
    //gpioSetValue(TOP, off);
    //usleep(200000);         // off for 200ms
    //cout << "Setting the TOP on" << endl;
    gpioSetValue(TOP, on);
    usleep(200000);            // 多次测算最佳结果        
    // cout << "TOP active finished." << endl;
    gpioSetValue(TOP, off);
    //usleep(200000);         // off for 200msss
    //gpioUnexport(TOP);     // unexport the TOP
    return 0;
}


