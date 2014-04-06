//
//  traceObj.cpp
//  capcam
//
//  Created by 沈 一鸣 on 14-4-4.
//  Copyright (c) 2014年 SYM. All rights reserved.
//

#include "traceObj.h"
#include "cppTweener.h"
#include <opencv2/opencv.hpp>
#include <string.h>
#include <chrono>
#include <future>

using namespace std;
using namespace cv;

tween::Tweener tweener;

static CvScalar colors[] = {
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
    return true;
}

void traceObj::update(){
    if (objects.empty() && goalObjects.empty()) {
        return; // both empty means traceObj detect not envoked
    }
    tweener.step(cvGetTickCount()/cvGetTickFrequency()*1000);
	if (drawMat){
		for (vector<Rect>::iterator r = objects.begin(); r != objects.end(); r++){
			rectangle(overLay, *r, colors[r - objects.begin()]);
		}
	}
	return;
};

void traceObj::detect(){
    if (!inited || srcFrame.empty()){
        return; // exit when not inited or empty input
    }
	
    double scale = 1.1;
    Mat smallFrame(srcFrame.size(), CV_8UC1);
	cvtColor(srcFrame, smallFrame, CV_BGR2GRAY);
//	resize(smallFrame, smallFrame, Size(256, 256));
	equalizeHist(smallFrame, smallFrame);

	double t = (double)cvGetTickCount(); // start evaluating process time
	cascade.detectMultiScale(smallFrame, goalObjects,scale,3,
					CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_CANNY_PRUNING,
					Size(50,50), Size(200,200) );
	t = (double)cvGetTickCount() - t;
    printf("detection time = %gms\n", t / ((double)cvGetTickFrequency()*1000.));
	addTween(); // Add tween animation to objects
	return;
};

void traceObj::addTween(){
	sortObj(goalObjects);
	for (vector<int>::size_type i = 0;i != goalObjects.size(); i++){
		int x, y, w, h;
		x = goalObjects[i].x;
		y = goalObjects[i].y;
		w = goalObjects[i].width;
		h = goalObjects[i].height;
	}
};