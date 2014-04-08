//
//  traceObj.cpp
//  capcam
//
//  Created by 沈 一鸣 on 14-4-4.
//  Copyright (c) 2014年 SYM. All rights reserved.
//

#include "traceObj.h"
#include <opencv2/opencv.hpp>
#include <string.h>
#include <chrono>
#include <future>

using namespace std;
using namespace cv;

const CvScalar traceObj::colors[] = {
        { { 0, 0, 255 } },
        { { 0, 128, 255 } },
        { { 0, 255, 255 } },
        { { 0, 255, 0 } },
        { { 255, 128, 0 } },
        { { 255, 255, 0 } },
        { { 255, 0, 0 } },
        { { 255, 0, 255 } }
    };

int sampleRate = 200;

string cascade_name = "data/haarcascade_frontalface_alt.xml";

bool traceObj::init(){
    // Initialize cascade system
    cascade.load(cascade_name);
    if (cascade.empty()) {
        cerr << "ERROR: Could not load classifier cascade:" << cascade_name << endl;
        return false; // End init with error
    }
    inited = true;
    started = true;
    return true;
}

void traceObj::attachFrame(cv::Mat &frame){
    srcFrame = &frame;
    if (drawMat) {
        overLay.create(frame.size(), CV_8UC3);
        overLay = cv::Scalar(0,0,0);
    }
    thread t(&traceObj::detect, this);
    t.detach();
};

void traceObj::update(){
    if (goalObjects.empty()) {
        return; // both empty means traceObj detect not envoked
    }
//    tweener.step(cvGetTickCount()/cvGetTickFrequency()*1000);
    step = ((double)(cvGetTickCount()-lastTick))/(double)cvGetTickFrequency()/1000./sampleRate;
    objects.resize(goalObjects.size());
    for (vector<Point2f>::size_type i = 0; i != goalObjects.size(); i++){
        Point2f Pt = pool[i].pos[1] + (pool[i].pos[0] - pool[i].pos[1] ) * step;
        float r = pool[i].radius[1] + (pool[i].radius[0] - pool[i].radius[1]) * step;
        objects[i].x = Pt.x - r/2;
        objects[i].y = Pt.y - r/2;
        objects[i].width = objects[i].height = r;
        if (drawMat) {
//            rectangle(overLay, objects[i], colors[i%8]);
            rectangle(overLay, goalObjects[i], colors[i%8]);
        }
    }
	return;
};

void traceObj::stop(){
    started = false;
    inited = false;
    // clear memory
    objects.clear();
    pool.clear();
};

void traceObj::detect(){
    while (started) {
        // wait sampleRate ms
        this_thread::sleep_for(std::chrono::milliseconds(sampleRate));
        
        if (!inited){
            continue; // exit when not inited or empty input
        }
        
        // do object detection
        double scale = 1.1;
        Mat smallFrame(srcFrame->size(), CV_8UC1);
        cvtColor(*srcFrame, smallFrame, CV_BGR2GRAY);
    //	resize(smallFrame, smallFrame, Size(256, 256));
        equalizeHist(smallFrame, smallFrame);

        double t = (double)cvGetTickCount(); // start evaluating process time
        cascade.detectMultiScale(smallFrame, goalObjects,scale,3,
                        CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_CANNY_PRUNING,
                        Size(50,50), Size(200,200) );
        t = (double)cvGetTickCount() - t;
        printf("detection time = %gms\n", t / ((double)cvGetTickFrequency()*1000.));
        if(pool.size()<goalObjects.size())
            pool.resize(goalObjects.size());
        for (vector<Rect>::size_type i = 0; i!= goalObjects.size(); i++) {
            for (int j = 9; j > 0; j--) {
                pool[i].pos[j] = pool[i].pos[j-1];
                pool[i].radius[j] = pool[i].radius[j-1];
            }
            pool[i].pos[0].x = goalObjects[i].x + goalObjects[i].width/2;
            pool[i].pos[0].y = goalObjects[i].y + goalObjects[i].height/2;
            pool[i].radius[0] = goalObjects[i].width;
        }
        lastTick = cvGetTickCount();
    }
	return;
};


//void traceObj::sortObj(vector<Rect> &obj){
//    return;
//};
