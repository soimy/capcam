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
#include <opencv2/opencv.hpp>
#include <string.h>
#include <chrono>
#include <future>

using namespace std;
using namespace cv;

//struct tmpObj {
//    float x, y, width, height;
//};

struct pointPool {
    Point2f pos[10];
    float radius[10];
};

class traceObj {
private:
    bool inited;
//	void sortObj(vector<Rect>&);
    Mat* srcFrame;
//  tween::Tweener tweener;
//  vector<float> x,y,r;
    int64 lastTick;
    double step;
    
public:
    vector<Rect> goalObjects;
    vector<Rect> objects;
    vector<pointPool> pool;
    string cascade_name;
    CascadeClassifier cascade;
	Mat overLay;
    int sampleRate; //in ms
	bool drawMat;
    bool started;
    static const CvScalar colors[];

    traceObj(){
        inited = false;
		drawMat = true;
        sampleRate = 200;
        started = false;
    };
    ~traceObj(){};
    bool init();
    void attachFrame(Mat&);
    void update();
    void detect();
    void stop();
};

#endif /* defined(__capcam__traceObj__) */

