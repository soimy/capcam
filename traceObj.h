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

private:
    static CvScalar colors[];
    bool inited;
	void addTween();
	void sortObj(vector<Rect>& obj);

public:
    vector<Rect> goalObjects;
    vector<Rect> objects;
    string cascade_name;
    static CascadeClassifier cascade;
    Mat srcFrame;
	Mat overLay;
    int sampleRate; //in ms
	bool drawMat;

    traceObj(){
        inited = false;
		drawMat = true;
    };
    ~traceObj(){};
    bool init();
    void update();
    void detect();
};

#endif /* defined(__capcam__traceObj__) */

