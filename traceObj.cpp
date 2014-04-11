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
#include <cmath>

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

// 15 times faster than the classical float sqrt.
// Reasonably accurate up to root(32500)
// Source: http://supp.iar.com/FilesPublic/SUPPORT/000419/AN-G-002.pdf
unsigned int fastSqrt(unsigned int x){
    unsigned int a,b;
    b = x;
    a = x = 0x3f;
    x = b/x;
    a = x = (x+a)>>1;
    x = b/x;
    a = x = (x+a)>>1;
    x = b/x;
    x = (x+a)>>1;
    return x;
}

unsigned int calcDist(Point2f a, Point2f b){
    unsigned int x,y;
    x = (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
    y = fastSqrt(x);
    return y;
    
}

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
}

void traceObj::update(){
    overLay = Scalar(0,0,0);
    if (pool.empty()) {
        return; // both empty means traceObj detect not envoked
    }
    // animationStep control animation position of object
    float animationStep;
    if(useAnimation)
        animationStep = ((double)(cvGetTickCount()-lastTick))/(double)cvGetTickFrequency()/1000./sampleRate;
    else
        animationStep = 1;
    
    objects.clear();
    for (vector<pointPool>::size_type i = 0; i != pool.size(); i++){
        
        // if pool member unstable, skip output
        if (pool[i].step >= 10)
            continue;
        
        Rect Rec0 = pool[i].rec[0];
        Rect Rec1 = pool[i].rec[1];
        Rect aniRec;
        Point2i tl0,tl1,wh0,wh1;
        Point2i tl,wh;
        tl0 = Point(Rec0.x,Rec0.y);
        tl1 = Point(Rec1.x,Rec1.y);
        wh0 = Point(Rec0.width,Rec0.height);
        wh1 = Point(Rec1.width,Rec1.height);

        tl = tl1+(tl0-tl1)*animationStep;
        wh = wh1+(wh0-wh1)*animationStep;
        
        aniRec = Rect(tl,Size(wh));
//        Point2i Pt1 = pool[i].pos[1];
//        Point2i Pt0 = pool[i].pos[0];
//        float r1 = pool[i].radius[1];
//        float r0 = pool[i].radius[0];
//        Point2i Pt = Pt1 + ( Pt0 - Pt1 ) * animationStep;
//        unsigned int r = r1 + ( r0 - r1 ) * animationStep;
//        Pt = Pt - Point(r/2,r/2);
        objects.push_back(aniRec);
        if (drawMat && useAnimation) {
                rectangle(*srcFrame, objects[i], colors[i%8]);
            if(drawTrack)
                for (int j = 0; j < 9; j++) {
                    // if pool cached pos = (0,0) means no cache, stop drawing track
                    if(pool[i].pos[j+1] == Point(0,0))
                        break;
                    cv::line(*srcFrame, pool[i].pos[j], pool[i].pos[j+1], Scalar(255-j*20, 0, 0));
                }
            if(drawId){
                cv::putText(*srcFrame, "ID: "+to_string(i)+" | step: "+to_string(pool[i].step), tl - Point(0,5), cv::FONT_HERSHEY_PLAIN, 0.8, colors[i%8]);
            }
        }
    }
    if (drawMat && !useAnimation) {
        for (vector<Rect>::iterator r=goalObjects.begin(); r!=goalObjects.end(); r++) {
            rectangle(*srcFrame, *r, colors[(r-goalObjects.begin())%8]);
            if(drawId){
                cv::putText(*srcFrame, "ID: "+to_string((r-goalObjects.begin())), Point(r->x,r->y) - Point(0,5), cv::FONT_HERSHEY_PLAIN, 0.8, colors[(r-goalObjects.begin())%8]);
            }
        }
    }
    cv::putText(*srcFrame, "Pool: "+to_string(pool.size()), Point(5,20), cv::FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255));
	return;
}

void traceObj::stop(){
    started = false;
    inited = false;
    // clear memory
    goalObjects.clear();
    objects.clear();
    pool.clear();
}

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
//      resize(smallFrame, smallFrame, Size(256, 256));
        equalizeHist(smallFrame, smallFrame);

//      double t = (double)cvGetTickCount(); // start evaluating process time
        cascade.detectMultiScale(smallFrame, goalObjects,scale,3,
                        CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_CANNY_PRUNING,
                        Size(100,100), Size(200,200) );
        pushPool(goalObjects);
//      t = (double)cvGetTickCount() - t;
//      printf("detection time = %gms\n", t / ((double)cvGetTickFrequency()*1000.));
        lastTick = cvGetTickCount();
    }
	return;
}



// Core function to compare and place trackId into Pool cache
void traceObj::pushPool(vector<Rect> obj){
    
    Mat tmpFrame = *srcFrame;
    
    // iterate through all pool member to find proper new trace obj
    for (vector<pointPool>::size_type i = 0; i!= pool.size(); i++) {
        
        // Check if pool member's step > 15, if true, delete it
        if (pool[i].step > 50) {
            pool.erase(pool.begin()+i);
            i--;
            continue;
        }
        
        bool cooked = false;
        // iterate through all new trace objs to find most fit pool member
        for (vector<Rect>::size_type m = 0; m!= obj.size(); m++){
            Point2f objPos;
            float objr;
            unsigned int newDist;
            
            objPos.x = obj[m].x + obj[m].width/2;
            objPos.y = obj[m].y + obj[m].height/2;
            objr = obj[i].width;
            
            newDist = calcDist(pool[i].pos[0], objPos);
            
            // if newDist is acceptable, go for graphics compare
            // expand distance check for greater step value
            int delta = newDist - pool[i].avgDist;
            delta = abs(delta);
            unsigned int rad = pool[i].radius[0];
            if (delta < rad) {
                
                // if graphics compare match, start pushing to pool
                if (matComp(tmpFrame(obj[m]).clone(), pool[i].trackId) > 0.7) {
                    for (int j = 9; j > 0; j--) {
                        pool[i].pos[j] = pool[i].pos[j-1];
                        pool[i].radius[j] = pool[i].radius[j-1];
                        pool[i].rec[j] = pool[i].rec[j-1];
                    }
                    pool[i].pos[0] = objPos;
                    pool[i].radius[0] = objr;
                    pool[i].rec[0] = obj[m];
                    pool[i].avgDist = pool[i].avgDist*.5 + newDist *.5;
                    
                    if(pool[i].step >1)
                        pool[i].step --; // important to set step to 1 for success trace
                    // copy the newly traced img to trackId
                    pool[i].trackId = tmpFrame(obj[m]).clone();
                    
                    // Remove from trace obj list to avoid duplicatation
                    // break the trace obj iteration, continue to next pool member
                    obj.erase(obj.begin() + m);
                    cooked = true; // match trace item sucessfully
                    break;
                }
            }
        }
        if(!cooked){
            // Copy last status to current status
            for (int j = 9; j > 0; j--) {
                pool[i].pos[j] = pool[i].pos[j-1];
                pool[i].radius[j] = pool[i].radius[j-1];
                pool[i].rec[j] = pool[i].rec[j-1];
            }
            // if no proper trace obj found, pool member go unstable
            pool[i].step++;
            
        }
    }
    
    // Check if there r still trace obj remain unmatched
    // If so, add pool member to be assigned
    if (!obj.empty()) {
        for (vector<Rect>::iterator r=obj.begin(); r!=obj.end(); r++) {
            pointPool newPool;
            Point2i objPos;
            unsigned int objr;
            
            objPos.x = r->x + r->width/2;
            objPos.y = r->y + r->height/2;
            objr = r->width;
            // Add current trace obj attr to new pool member
            for(int i=0; i<10; i++){
                newPool.pos[i] = Point(0,0);
                newPool.radius[i] = 0;
                newPool.rec[i] = Rect(Point(0,0),Size(0,0));
            }
            newPool.pos[0] = objPos;
            newPool.radius[0] = objr;
            newPool.rec[0] = *r;
            newPool.step = 11; // activate step and set to unstable
            newPool.trackId = tmpFrame(*r).clone(); // copy the traced img to trackId
            
            // add to pool
            pool.push_back(newPool);
        }
    }
//    cout << "Traced Objects' pool size: " << pool.size() << endl;
    return;
}

float traceObj::matComp(cv::Mat newId, cv::Mat srcId){
    return 1.;
}