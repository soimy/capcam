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

#define DETECT_CASCADE  0
#define DETECT_BLOB     1

#define USEANIM         1  // 2^0, bit 0
#define DRAWMAT         2  // 2^1, bit 1
#define DRAWTRACK       4  // 2^2, bit 2
#define DRAWID          8  // 2^3, bit 3


class traceObj {
public:
    // default constructor
    traceObj();
    // constructor with parameters
    traceObj(cv::Mat& _srcFrame, unsigned int _sampleRate, unsigned int _detect_flags, unsigned char _flags);
    // decontructor
    ~traceObj();
    // Settings memeber function
    void setSampleRate(unsigned int);
    void setFlags( unsigned int _detect_flags, unsigned char _flags);
    bool setCascade(string);
    void setBlobParam(vector<Vec3b>, Vec3b, float, float);
    void attachFrame(Mat&);
    // Runtime function
    bool init();
    bool init(Mat& _srcFrame, unsigned int _sampleRate, unsigned int _detect_flags, unsigned char _flags);
    void update();
    void detect();

private:
    struct pointPool {
        Point2i pos[10];
        Point2i stablizedPos[10];
        Point2i predicPos[10];
        unsigned int radius[10];
        unsigned int step; // 0 = deactivated, <2 stable, >5 unstable
        Mat trackId;
        KalmanFilter kfc;
    };
    unsigned char flags;        // flags for predifined settings
    unsigned int detect_flags;  // flags for type of detection
    bool started;               // status flag
    Mat *srcFrame;              // input frame pointer
    int64 lastTick;             // Time ticker
    vector<Rect> goalObjects;   // Raw detected rectangles
    vector<Rect> objects;       // Stablized trace rectangles
    vector<pointPool> pool;     // pool to cache all traced object
    string cascade_name;        // cascadeName for used with CascadeClassifier
    CascadeClassifier cascade;  // CascadeClassifier
    vector<Vec3b> keyColors;    // HSV color for tracing blobs
    float minArea, maxArea;     // blob min-max area threshold
    Vec3b hsvRange;             // HSV value range for color object keying
    int sampleRate;             // sample interval in ms
    static const CvScalar colors[];

    void pushPool(vector<Rect>);// Recognize and sit dectected objects to pool
    float matComp(Mat,Mat);     // func to compare 2 img, return fitting rate
    virtual void doCascadeDetect();
    virtual void doBlobDetect();
    void colorKey(Mat& src, Mat& dst, Vec3b keyColor, Vec3b hsvRange);


};

#endif /* defined(__capcam__traceObj__) */

