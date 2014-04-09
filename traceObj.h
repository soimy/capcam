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
    Point2i pos[10];
    float radius[10];
    unsigned int avgDist = 200;
    unsigned int step; // 0 = deactivated, <2 stable, >5 unstable
    Mat trackId;
};

class traceObj {
private:
    bool inited;
    cv::Mat* srcFrame;
    int64 lastTick;
    double animationStep;
    void pushPool(vector<Rect>);
    float matComp(Mat,Mat);
    
public:
    vector<Rect> goalObjects;
    vector<Rect> objects;
    vector<pointPool> pool;
    string cascade_name;
    CascadeClassifier cascade;
	Mat overLay;
    int sampleRate; //in ms
	bool drawMat;
    bool drawTrack;
    bool drawId;
    bool started;
    static const CvScalar colors[];

    traceObj(){
        inited = false;
		drawMat = true;
        drawTrack = true;
        drawId = true;
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

