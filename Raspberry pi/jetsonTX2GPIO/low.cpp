// low.c

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

    //cout << "active LOW" << endl;
    jetsonTX2GPIONumber LOW = gpio255 ;     // Ouput
    // Make the LOW available in user space
    //gpioExport(LOW) ;
    //gpioSetDirection(LOW,outputPin) ;
    //cout << "Setting the LOW off" << endl;
    //gpioSetValue(LOW, off); // 255 gpio口初始化为零
    //usleep(50000);         // off for 200ms
    //cout << "Setting the LOW on" << endl;
    gpioSetValue(LOW, on);
    usleep(12000);         // off for 11111us
    //cout << "LOW active finished." << endl;
    gpioSetValue(LOW, off);
    //usleep(200000);         // off for 200ms
    //gpioUnexport(LOW);     
    return 0;
}


