//
//  traceObj.cpp
//  capcam
//
//  Created by 沈 一鸣 on 14-4-4.
//  Copyright (c) 2014年 SYM. All rights reserved.
//

#include "traceObj.h"
#include "munkres.h"

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

float legacy; // for detection CPU time caculator
unsigned long trackerCounter; // for tracker counter

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

void drawCross(Mat& img, Point2i center, Scalar crossColor, int d){
    line(img, Point(center.x-d, center.y-d), Point(center.x+d, center.y+d), crossColor, 1, CV_AA, 0);
    line(img, Point(center.x+d, center.y-d), Point(center.x-d, center.y+d), crossColor, 1, CV_AA, 0);
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
    // animationStep control animation position of object
    float animationStep;
    if(useAnimation)
        animationStep = ((double)(cvGetTickCount()-lastTick))/(double)cvGetTickFrequency()/1000./sampleRate;
    else
        animationStep = 1;
    
    objects.clear();
    for (vector<pointPool>::size_type i = 0; i != pool.size(); i++){
        
        // if pool member unstable, skip output
//        if (pool[i].step >= 10 ) continue;
        
        Point2i Pt1 = pool[i].stablizedPos[1];
        Point2i Pt0 = pool[i].stablizedPos[0];
        float r1 = pool[i].radius[1];
        float r0 = pool[i].radius[0];
        Point2i Pt = Pt1 + ( Pt0 - Pt1 ) * animationStep;
        unsigned int r = r1 + ( r0 - r1 ) * animationStep;
        Pt = Pt - Point(r/2,r/2);
        objects.push_back(Rect(Pt,Size(r,r)));
        if (drawMat && useAnimation) {
                rectangle(*srcFrame, objects[i], colors[i%8]);
            if(drawTrack){
                for (int j = 0; j < 9; j++) {
                    // if pool cached pos = (0,0) means no cache, stop drawing track
                    if(pool[i].pos[j+1] == Point(0,0))
                        break;
                    // Draw detected track
                    drawCross(*srcFrame, pool[i].pos[j], Scalar(100, 100, 255-j*20), 3);
                    cv::line(*srcFrame, pool[i].pos[j], pool[i].pos[j+1], Scalar(100, 100, 255-j*20));
                    // Draw kalman filterd track
                    cv::line(*srcFrame, pool[i].stablizedPos[j], pool[i].stablizedPos[j+1], Scalar(100, 255-j*20, 255-j*20));
                }
                // draw current detected Point
                drawCross(*srcFrame, pool[i].pos[0], Scalar(100,100,255), 5);
                // draw estimated point
                drawCross(*srcFrame, pool[i].stablizedPos[0], Scalar(100,255,255), 5);
                drawCross(*srcFrame, pool[i].predicPos[0], Scalar(0,0,255), 5);
            }
            if(drawId){
                cv::putText(*srcFrame, "ID: "+to_string(i)+" | step: "+to_string(pool[i].step), Pt - Point(0,5), cv::FONT_HERSHEY_PLAIN, 0.8, colors[i%8]);
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
    cv::putText(*srcFrame, "Pool: "+to_string(pool.size()) +" Detect Lag: "+to_string((int)legacy+sampleRate)+" ms Tracker: "+to_string(trackerCounter), Point(5,20), cv::FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255));
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

        double t = (double)cvGetTickCount(); // start evaluating process time
        cascade.detectMultiScale(smallFrame, goalObjects,scale,3,
                        CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_CANNY_PRUNING,
                        Size(100,100), Size(200,200) );
        pushPool(goalObjects);
        t = (double)cvGetTickCount() - t;
//        printf("detection time = %gms\n", t / ((double)cvGetTickFrequency()*1000.));
        legacy = t / ((double)cvGetTickFrequency()*1000.);
        lastTick = cvGetTickCount();
        trackerCounter = goalObjects.size();
    }
	return;
}



// Core function to compare and place trackId into Pool cache
void traceObj::pushPool(vector<Rect> obj){
    
    Mat tmpFrame = *srcFrame;
    
    // Remove all non-activated pool member
    for (vector<pointPool>::size_type i=0; i!= pool.size(); i++) {
        if (pool[i].step > 20) {
            pool.erase(pool.begin()+i);
            i--;
        }
    }
    
//    vector< vector<int> > delta_matrix;
//    vector< vector<int> > assign_matrix;
    int rows = (int)pool.size();
    int cols = (int)obj.size();
    Mat_<int> delta_matrix(rows,cols);
//    delta_matrix.resize(pool.size(), vector<int>(obj.size(),0));
//    assign_matrix.resize(pool.size(), vector<int>(obj.size(),0));
    
    
    // iterate through all pool member to find proper new trace obj
    for (vector<pointPool>::size_type i = 0; i!= pool.size(); i++) {
        
//        int smallestDelta = 1000;
//        vector<Rect>::size_type smallestIndex = 0;
        Point2f objPos;
        float objr;
        unsigned int newDist;//, smallDist = 50;
        
        // kalman prediction setup
        Mat prediction = pool[i].kfc.predict();
        Point2i predictPt;
        if(pool[i].predicPos[5].x == 0)
            predictPt = pool[i].pos[0];
        else
            predictPt = Point(prediction.at<float>(0),prediction.at<float>(1));
        
        // Push predicted point to pool cache
        for (int k = 9; k > 0; k--)
            pool[i].predicPos[k] = pool[i].predicPos[k-1];
        pool[i].predicPos[0] = predictPt;

        // iterate through all new trace objs to find most fit pool member
        for (vector<Rect>::size_type m = 0; m!= obj.size(); m++){

            objPos.x = obj[m].x + obj[m].width/2;
            objPos.y = obj[m].y + obj[m].height/2;
            objr = obj[m].width;
        
            newDist = calcDist(predictPt, objPos);
            delta_matrix((unsigned int)i,(unsigned int)m) = newDist;
            
            // if newDist is acceptable, go for graphics compare
            // expand distance check for greater step value
//            int delta = newDist - pool[i].avgDist;
//            delta = abs(delta);
//            if(delta < smallestDelta){
//                smallestDelta = delta;
//                smallestIndex = m;
//                smallDist = newDist;
//            }

        }
    }
    
    // Solve Hungarian assignment
    Munkres m;
    m.solve(delta_matrix);
    
//    if (smallestDelta < pool[i].radius[0] * 0.6) { // delta smaller than side by side detect rectangle
    
    for( int i=0; i!=rows; i++){
        
        Mat estimated;
        Point2i stablizedPt;
        Mat_<float> measurement(3,1);
        
        Point2i objPos;
        unsigned int objr;
        bool assigned = false;
        
        // iterate through current columne to find proper assignment
        for (int j=0; j<cols; j++) {
            // skip  not assigned
            if(delta_matrix(i,j) != 0)
                continue;
            
            // if graphics compare match, start pushing to pool
            if (matComp(tmpFrame(obj[j]).clone(), pool[i].trackId) > 0.7) {
                // Shift the pool position
                for (int k = 9; k > 0; k--) {
                    pool[i].pos[k] = pool[i].pos[k-1];
                    pool[i].stablizedPos[k] = pool[i].stablizedPos[k-1];
                    pool[i].predicPos[k] = pool[i].predicPos[k-1];
                    pool[i].radius[k] = pool[i].radius[k-1];
                    pool[i].rec[k] = pool[i].rec[k-1];
                }
                
                objPos.x = obj[j].x + obj[j].width/2;
                objPos.y = obj[j].y + obj[j].height/2;
                objr = obj[j].width;
                
                
                // evaluate kalman prediction
                //
                measurement(0) = objPos.x;
                measurement(1) = objPos.y;
                measurement(2) = objr;
                
                estimated = pool[i].kfc.correct(measurement);
                stablizedPt = Point(estimated.at<float>(0),estimated.at<float>(1));
                
                
                pool[i].pos[0] = objPos;
                pool[i].stablizedPos[0] = stablizedPt;
                pool[i].radius[0] = (int)estimated.at<float>(2);
                pool[i].rec[0] = obj[j];
//                pool[i].avgDist = pool[i].avgDist*.75 + smallDist *.25;
                
                if(pool[i].step >1)
                    pool[i].step --; // important to set step to 1 for success trace
                // copy the newly traced img to trackId
                pool[i].trackId = tmpFrame(obj[j]).clone();
                
                // Remove from trace obj list to avoid duplicatation
//                obj.erase(obj.begin() + j);
                // break the trace obj iteration, continue to next pool member
                assigned = true;
                break; // finish assignment, end the iteration
            }
        }
        // If no proper assignment found
        // Copy last status to current status
        if(!assigned){
            for (int k = 9; k > 0; k--) {
                pool[i].pos[k] = pool[i].pos[k-1];
                pool[i].stablizedPos[k] = pool[i].stablizedPos[k-1];
                pool[i].radius[k] = pool[i].radius[k-1];
                pool[i].rec[k] = pool[i].rec[k-1];
            }
            // If coresponding detected position is missing,
            // Set estimated Position to current Position
            Point2i predictPt = pool[i].predicPos[0];
            if(predictPt.x != 0){
                measurement(0) = predictPt.x;
                measurement(1) = predictPt.y;
                measurement(2) = pool[i].radius[0];
                pool[i].pos[0] = predictPt;
            }else{
                measurement(0) = pool[i].pos[0].x;
                measurement(1) = pool[i].pos[0].y;
                measurement(2) = pool[i].radius[0];
                // pool[i].pos[0] remain untouched
            }
            estimated = pool[i].kfc.correct(measurement);
            stablizedPt = Point(estimated.at<float>(0),estimated.at<float>(1));
            pool[i].stablizedPos[0] = stablizedPt;
            pool[i].predicPos[0] = predictPt;
            pool[i].radius[0] = estimated.at<float>(2);
            // if no proper trace obj found, pool member go unstable
            pool[i].step++;
        }
        
    }
    
    
    // iterate through columns to find un-assigned tracker
    for (int j=0; j<cols; j++) {
        bool tag = true;
        for (int i=0; i<rows; i++) {
            if(delta_matrix(i,j) == 0)
                tag = false;
        }
        if (tag) {
            
            // insert new tracker to pool
            // and do pool member initializing
            //
            for (vector<Rect>::iterator r=obj.begin(); r!=obj.end(); r++) {
                pointPool newPool;
                Point2i objPos;
                unsigned int objr;
                
                objPos.x = r->x + r->width/2;
                objPos.y = r->y + r->height/2;
                objr = r->width;
                
                newPool.pos[0] = objPos;
                newPool.stablizedPos[0] = objPos;
                newPool.predicPos[0] = objPos;
                newPool.radius[0] = objr;
                newPool.rec[0] = *r;
                newPool.step = 15; // activate step and set to unstable
                newPool.trackId = tmpFrame(*r).clone(); // copy the traced img to trackId
                
                // Initialize the Kalman filter for position prediction
                //
                newPool.kfc.init(6, 3, 0);
                // Setup transitionMatrix to
                // 1, 0, 1, 0
                // 0, 1, 0, 1
                // 0, 0, 1, 0
                // 0, 0, 0, 1
                // very weird, don't understand ....
                newPool.kfc.transitionMatrix = *(Mat_<float>(6,6)
                                            <<  1,0,0,3,0,0,
                                                0,1,0,0,3,0,
                                                0,0,1,0,0,1,
                                                0,0,0,1,0,0,
                                                0,0,0,0,1,0,
                                                0,0,0,0,0,1);
                
                newPool.kfc.statePost.at<float>(0) = objPos.x;
                newPool.kfc.statePost.at<float>(1) = objPos.y;
                newPool.kfc.statePost.at<float>(2) = objr;
                newPool.kfc.statePost.at<float>(3) = 0;
                newPool.kfc.statePost.at<float>(4) = 0;
                newPool.kfc.statePost.at<float>(5) = 0;
                
                newPool.kfc.statePre.at<float>(0) = objPos.x;
                newPool.kfc.statePre.at<float>(1) = objPos.y;
                newPool.kfc.statePre.at<float>(2) = objr;
                newPool.kfc.statePre.at<float>(3) = 0;
                newPool.kfc.statePre.at<float>(4) = 0;
                newPool.kfc.statePre.at<float>(5) = 0;
                
                setIdentity(newPool.kfc.measurementMatrix);
                setIdentity(newPool.kfc.processNoiseCov, Scalar::all(1e-2));
                setIdentity(newPool.kfc.measurementNoiseCov, Scalar::all(1e-1));
                setIdentity(newPool.kfc.errorCovPost, Scalar::all(0.1));
                
                // add to pool
                pool.push_back(newPool);
            }

        }
        
    }
    return;
}

float traceObj::matComp(cv::Mat newId, cv::Mat srcId){
    return 1.; //Just return true by now, will do some more work future
}