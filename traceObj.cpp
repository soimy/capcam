//
//  traceObj.cpp
//  capcam
//
//  Created by 沈 一鸣 on 14-4-4.
//  Copyright (c) 2014年 SYM. All rights reserved.
//

#include "traceObj.h"
#include "hungarian.h"
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
        if (pool[i].step >= 5 )
            continue;
        
//        Rect Rec0 = pool[i].rec[0];
//        Rect Rec1 = pool[i].rec[1];
//        Rect aniRec;
//        Point2i tl0,tl1,wh0,wh1;
//        Point2i tl,wh;
//        tl0 = Point(Rec0.x,Rec0.y);
//        tl1 = Point(Rec1.x,Rec1.y);
//        wh0 = Point(Rec0.width,Rec0.height);
//        wh1 = Point(Rec1.width,Rec1.height);
//
//        tl = tl1+(tl0-tl1)*animationStep;
//        wh = wh1+(wh0-wh1)*animationStep;
//        
//        aniRec = Rect(tl,Size(wh));
//        objects.push_back(aniRec);
        
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
    cv::putText(*srcFrame, "Pool="+to_string(pool.size()) +" | Detect Lag="+to_string((int)legacy+sampleRate)+" ms", Point(5,20), cv::FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255));
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
    
    vector< vector<int> > delta_matrix;
    vector< vector<int> > assign_matrix;
    delta_matrix.resize(pool.size(), vector<int>(obj.size(),0));
    assign_matrix.resize(pool.size(), vector<int>(obj.size(),0));
    
    
    // iterate through all pool member to find proper new trace obj
    for (vector<pointPool>::size_type i = 0; i!= pool.size(); i++) {
        
        int smallestDelta = 1000;
        vector<Rect>::size_type smallestIndex = 0;
        Point2f objPos;
        float objr;
        unsigned int newDist, smallDist = 50;
        
        // kalman prediction setup
        Mat prediction = pool[i].kfc.predict();
        Point2i predictPt;
        if(pool[i].predicPos[5].x == 0)
            predictPt = pool[i].pos[0];
        else
            predictPt = Point(prediction.at<float>(0),prediction.at<float>(1));
        
        Mat estimated;
        Point2i stablizedPt;
        Mat_<float> measurement(2,1);

        // iterate through all new trace objs to find most fit pool member
        for (vector<Rect>::size_type m = 0; m!= obj.size(); m++){

            objPos.x = obj[m].x + obj[m].width/2;
            objPos.y = obj[m].y + obj[m].height/2;
            objr = obj[m].width;
        
            newDist = calcDist(pool[i].pos[0], objPos);
            
            // if newDist is acceptable, go for graphics compare
            // expand distance check for greater step value
            int delta = newDist - pool[i].avgDist;
            delta = abs(delta);
            if(delta < smallestDelta){
                smallestDelta = delta;
                smallestIndex = m;
                smallDist = newDist;
            }

        }
        if (smallestDelta < pool[i].radius[0] * 0.6) { // delta smaller than side by side detect rectangle
            
            
          
            // if graphics compare match, start pushing to pool
            if (matComp(tmpFrame(obj[smallestIndex]).clone(), pool[i].trackId) > 0.7) {
                // Shift the pool position
                for (int j = 9; j > 0; j--) {
                    pool[i].pos[j] = pool[i].pos[j-1];
                    pool[i].stablizedPos[j] = pool[i].stablizedPos[j-1];
                    pool[i].predicPos[j] = pool[i].predicPos[j-1];
                    pool[i].radius[j] = pool[i].radius[j-1];
                    pool[i].rec[j] = pool[i].rec[j-1];
                }
                
                objPos.x = obj[smallestIndex].x + obj[smallestIndex].width/2;
                objPos.y = obj[smallestIndex].y + obj[smallestIndex].height/2;
                objr = obj[smallestIndex].width;
                
                
                
                // evaluate kalman prediction
                //
                measurement(0) = objPos.x;
                measurement(1) = objPos.y;
                
                estimated = pool[i].kfc.correct(measurement);
                stablizedPt = Point(estimated.at<float>(0),estimated.at<float>(1));
                if(predictPt.x == 0)
                    predictPt = stablizedPt;
                
                
                
                
                pool[i].pos[0] = objPos;
                pool[i].stablizedPos[0] = stablizedPt;
                pool[i].predicPos[0] = predictPt;
                pool[i].radius[0] = objr;
                pool[i].rec[0] = obj[smallestIndex];
                pool[i].avgDist = pool[i].avgDist*.75 + smallDist *.25;
                
                if(pool[i].step >1)
                    pool[i].step --; // important to set step to 1 for success trace
                // copy the newly traced img to trackId
                pool[i].trackId = tmpFrame(obj[smallestIndex]).clone();
                
                // Remove from trace obj list to avoid duplicatation
                // break the trace obj iteration, continue to next pool member
                obj.erase(obj.begin() + smallestIndex);
            }
        }else{
            // Copy last status to current status
            for (int j = 9; j > 0; j--) {
                pool[i].pos[j] = pool[i].pos[j-1];
                pool[i].stablizedPos[j] = pool[i].stablizedPos[j-1];
                pool[i].predicPos[j] = pool[i].predicPos[j-1];
                pool[i].radius[j] = pool[i].radius[j-1];
                pool[i].rec[j] = pool[i].rec[j-1];
            }
            // If coresponding detected position is missing,
            // Set estimated Position to current Position
            if(predictPt.x != 0){
                measurement(0) = predictPt.x;
                measurement(1) = predictPt.y;
                pool[i].pos[0] = predictPt;
            }else{
                measurement(0) = pool[i].pos[0].x;
                measurement(1) = pool[i].pos[0].y;
                // pool[i].pos[0] remain untouched
            }
            estimated = pool[i].kfc.correct(measurement);
            stablizedPt = Point(estimated.at<float>(0),estimated.at<float>(1));
            pool[i].stablizedPos[0] = stablizedPt;
            pool[i].predicPos[0] = predictPt;
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
            
            newPool.pos[0] = objPos;
            newPool.stablizedPos[0] = objPos;
            newPool.predicPos[0] = objPos;
            newPool.radius[0] = objr;
            newPool.rec[0] = *r;
            newPool.step = 11; // activate step and set to unstable
            newPool.trackId = tmpFrame(*r).clone(); // copy the traced img to trackId
            
            // Initialize the Kalman filter for position prediction
            //
            newPool.kfc.init(4, 2, 0);
            // Setup transitionMatrix to
            // 1, 0, 1, 0
            // 0, 1, 0, 1
            // 0, 0, 1, 0
            // 0, 0, 0, 1
            // very weird, don't understand ....
            newPool.kfc.transitionMatrix = *(Mat_<float>(4,4) << 1,0,1,0, 0,1,0,1, 0,0,1,0, 0,0,0,1);
            newPool.kfc.statePost.setTo(Scalar(objPos.x,objPos.y));
            newPool.kfc.statePre.at<float>(0) = objPos.x;
            newPool.kfc.statePre.at<float>(1) = objPos.y;
            newPool.kfc.statePre.at<float>(2) = 0;
            newPool.kfc.statePre.at<float>(3) = 0;
            setIdentity(newPool.kfc.measurementMatrix);
            setIdentity(newPool.kfc.processNoiseCov, Scalar::all(1e-2));
            setIdentity(newPool.kfc.measurementNoiseCov, Scalar::all(1e-1));
            setIdentity(newPool.kfc.errorCovPost, Scalar::all(0.1));
            
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