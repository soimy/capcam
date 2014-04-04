//
//  traceObj.h
//  capcam
//
//  Created by 沈 一鸣 on 14-4-4.
//  Copyright (c) 2014年 SYM. All rights reserved.
//

#ifndef __capcam__traceObj__
#define __capcam__traceObj__

#include <iostream>
#include "cppTweener.h"
#include <opencv2/opencv.hpp>
#include <string.h>
#include <chrono>
#include <future>

using namespace std;
using namespace cv;

class traceObj : public tween::TweenerListener {
public:
    vector<Rect> goalObjects;
    vector<Rect> objects;
    string cascade_name;
    static CascadeClassifier cascade;
    Mat srcFrame;
    int sampleRate; //in ms
    
    traceObj(){
        inited = false;
    };
    ~traceObj(){};
    bool init();
    void update();
    void detect();
private:
    static CvScalar colors[];
    bool inited;
};

#endif /* defined(__capcam__traceObj__) */

