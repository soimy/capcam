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
#include <opencv2/features2d/features2d.hpp>
#include <string.h>
#include <chrono>
#include <future>
#include <cmath>
#include <thread>

using namespace std;
using namespace cv;

const CvScalar traceObj::colors[] = {
        { { 0,  0,  255 } },
        { { 0,  128,255 } },
        { { 0,  255,255 } },
        { { 0,  255,0 } },
        { { 255,128,0 } },
        { { 255,255,0 } },
        { { 255,0,  0 } },
        { { 255,0,  255 } }
    };

float legacy; // for detection CPU time caculator
unsigned long trackerCounter; // for tracker counter

///////////////////////////////////////////////
// Start of local usage caculating functions //
///////////////////////////////////////////////

// 15 times faster than the classical float sqrt.
// Reasonably accurate up to root(32500)
// Source: http://supp.iar.com/FilesPublic/SUPPORT/000419/AN-G-002.pdf
unsigned int
fastSqrt(unsigned int x){
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

unsigned int
calcDist(Point2f a, Point2f b){
    unsigned int x,y;
    x = (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
    y = fastSqrt(x);
    return y;
}

void
drawCross(Mat& img, Point2i center, Scalar crossColor, int d){
    line(img, Point(center.x-d, center.y-d), Point(center.x+d, center.y+d), crossColor, 1, CV_AA, 0);
    line(img, Point(center.x+d, center.y-d), Point(center.x-d, center.y+d), crossColor, 1, CV_AA, 0);
}

// 0..255 version curve interpolation maya like function
int
linstep(int _start, int _end, int _step){
    //return _step<_start?0:(_step>_end?255:(int)((_step-_start)/(_end-_start)*255.0f));
    if(_start >= _end)
        return 0;

    if(_step < _start)
        return 0;
    else if(_step > _end)
        return 255;
    else
        return (int)(((float)_step-(float)_start)/((float)_end-(float)_start)*255.0f);
}

int
isInside(int _start, int _end, int _step){
    if(_start >= _end)
        return 0;

    return _step<_start?0:(_step>_end?0:1);
}



/////////////////////////////////////////////////////
// Start of TraceObj class construction functions  //
/////////////////////////////////////////////////////

traceObj::traceObj(){
    sampleRate = 300;
    started = false;
    flags = USEANIM | DRAWMAT | DRAWTRACK | DRAWID ; // equals hex 0xf
    detect_flags = DETECT_CASCADE;
    cascade_name = "data/haarcascade_frontalface_alt.xml";
    keyColors.push_back(cv::Vec3b(11,119,255));
    hsvRange = Vec3b(10,200,220);
    minArea = 800.0f;
    maxArea = 100000.0f;
}

traceObj::traceObj(
        cv::Mat& _srcFrame,
        unsigned int _sampleRate = 300,
        unsigned int _detect_flags = DETECT_CASCADE,
        unsigned char _flags = 0xf ){

    sampleRate = _sampleRate;
    flags = _flags;
    detect_flags = _detect_flags;
    srcFrame = &_srcFrame;
    started = false;
    if(cascade_name.empty())
        cascade_name = "data/haarcascade_frontalface_alt.xml";
    keyColors.push_back(cv::Vec3b(11,119,255));
    hsvRange = Vec3b(10,200,220);
    minArea = 800.0f;
    maxArea = 100000.0f;
}

traceObj::~traceObj(){
    started = false;
    // wait for detached detection thread to stop
    this_thread::sleep_for( std::chrono::milliseconds(sampleRate));
    // clear memory
    goalObjects.clear();
    objects.clear();
    pool.clear();
}

////////////////////////////////////////////////
// Start of TraceObj class setting functions  //
////////////////////////////////////////////////

void
traceObj::setSampleRate(unsigned int _sampleRate){
   sampleRate = _sampleRate;
}

void
traceObj::setFlags(unsigned int _detect_flags, unsigned char _flags = 0xf){
    detect_flags = _detect_flags;
    flags = _flags;
    if(detect_flags == DETECT_CASCADE && cascade.empty())
        setCascade("data/haarcascade_frontalface_alt.xml");
}

bool
traceObj::setCascade(string _cascade_name){
    cascade_name = _cascade_name;
    cascade.load(cascade_name);
    if(cascade.empty()){
        cerr << "Error loading cascade: " << _cascade_name << endl;
        return false;
    }
    return true;
}

void
traceObj::setBlobParam(vector<Vec3b> _keyColors, Vec3b _hsvRange, float _minArea, float _maxArea){
    keyColors = _keyColors;
    hsvRange = _hsvRange;
    minArea = _minArea;
    maxArea = _maxArea;
}


void
traceObj::attachFrame(cv::Mat &frame){
    srcFrame = &frame;
    //thread t(&traceObj::detect, this);
    //t.detach();
}

////////////////////////////////////////////////
// Start of TraceObj class runtime functions  //
////////////////////////////////////////////////

bool
traceObj::init(){
    bool init_status;
    if(started){
        started = false;
        this_thread::sleep_for( std::chrono::milliseconds(sampleRate));
        // wait for independent detection thread to terminate
    }
    // Initialize cascade system
    if(detect_flags == DETECT_CASCADE)
        init_status = setCascade(cascade_name);
    else
        init_status = true;

    if(init_status)
        started = true;

    thread t(&traceObj::detect, this);
    t.detach();

    return init_status;
}

bool
traceObj::init(
        Mat& _srcFrame,
        unsigned int _sampleRate = 300,
        unsigned int _detect_flags = DETECT_CASCADE,
        unsigned char _flags = 0xf){
    bool init_status;
    if(started){
        started = false;
        this_thread::sleep_for( std::chrono::milliseconds(sampleRate));
        // wait for independent detection thread to terminate
    }
    srcFrame = &_srcFrame;
    sampleRate = _sampleRate;
    detect_flags = _detect_flags;
    flags = _flags;

    if(detect_flags == DETECT_CASCADE)
        init_status = setCascade(cascade_name);
    else
        init_status = true;

    if(init_status)
        started = true;

    thread t(&traceObj::detect, this);
    t.detach();

    return init_status;
}

void
traceObj::update(){
    // ignore process if not initialized
    if(!started)
        return;

    float animationStep; // animationStep control animation position of object
    if( (flags&USEANIM) == USEANIM )
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

        if( (flags&DRAWMAT) != DRAWMAT ) break;

        rectangle(*srcFrame, objects[i], colors[i%8]);

        if( (flags&DRAWTRACK) == DRAWTRACK ){
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
        if( (flags&DRAWID) == DRAWID ){
            cv::putText(*srcFrame, "ID: "+to_string(i)+" | step: "+to_string(pool[i].step), Pt - Point(0,5), cv::FONT_HERSHEY_PLAIN, 0.8, colors[i%8]);
        }
    }
    if( (flags&DRAWMAT) == DRAWMAT )
        cv::putText(*srcFrame,
                "Pool: "+to_string(pool.size()) +
                " Detect Lag: "+to_string((int)legacy+sampleRate)+
                " ms Tracker: "+to_string(trackerCounter),
                Point(5,20), cv::FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255));
	return;
}


void
traceObj::detect(){
    while (started) {
        // wait sampleRate ms
        this_thread::sleep_for(std::chrono::milliseconds(sampleRate));

        // Check if params are OK
        // if not, continue to next loop and check again
        if(srcFrame->empty()) continue;
        if(cascade.empty() && detect_flags == DETECT_CASCADE) continue;
        if(detect_flags == DETECT_BLOB && keyColors.empty()) continue;

        // do object detection
        double t = (double)cvGetTickCount(); // start evaluating process time

        if(detect_flags == DETECT_CASCADE)
            doCascadeDetect();
        else
            doBlobDetect();

        t = (double)cvGetTickCount() - t;
        legacy = t / ((double)cvGetTickFrequency()*1000.);
        lastTick = cvGetTickCount();
        trackerCounter = goalObjects.size();
    }
	return;
}


////////////////////////////////////////////////
// Start of TraceObj class private functions  //
////////////////////////////////////////////////

// Core function to compare and place trackId into Pool cache
void
traceObj::pushPool(vector<Rect> obj){

    Mat tmpFrame = *srcFrame;

    // Remove all non-activated pool member
    for (vector<pointPool>::size_type i=0; i!= pool.size(); i++) {
        if (pool[i].step > 20) {
            pool.erase(pool.begin()+i);
            i--;
        }
    }

    int rows = (int)pool.size();
    int cols = (int)obj.size();
    Mat_<int> delta_matrix(rows,cols);

    // iterate through all pool member
    // to calculate delta_matrix
    for (vector<pointPool>::size_type i = 0; i!= pool.size(); i++) {

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

        }
    }

    // Solve Hungarian assignment
    Munkres m;
    m.diag(true);
    m.solve(delta_matrix);

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
            //if (matComp(tmpFrame(obj[j]).clone(), pool[i].trackId) > 0.7) {
            // Shift the pool position
            for (int k = 9; k > 0; k--) {
                pool[i].pos[k] = pool[i].pos[k-1];
                pool[i].stablizedPos[k] = pool[i].stablizedPos[k-1];
                pool[i].predicPos[k] = pool[i].predicPos[k-1];
                pool[i].radius[k] = pool[i].radius[k-1];
                //pool[i].rec[k] = pool[i].rec[k-1];
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
            //pool[i].rec[0] = obj[j];
//                pool[i].avgDist = pool[i].avgDist*.75 + smallDist *.25;

            if(pool[i].step >1)
                pool[i].step --; // important to set step to 1 for success trace
            // copy the newly traced img to trackId
            //pool[i].trackId = tmpFrame(obj[j]).clone();

            // Remove from trace obj list to avoid duplicatation
//                obj.erase(obj.begin() + j);
            // break the trace obj iteration, continue to next pool member
            assigned = true;
            break; // finish assignment, end the iteration
        }
        // If no proper assignment found
        // Copy last status to current status
        if(!assigned){
            for (int k = 9; k > 0; k--) {
                pool[i].pos[k] = pool[i].pos[k-1];
                pool[i].stablizedPos[k] = pool[i].stablizedPos[k-1];
                pool[i].radius[k] = pool[i].radius[k-1];
                //pool[i].rec[k] = pool[i].rec[k-1];
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
    // and assign to new pool member
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
                //newPool.rec[0] = *r;
                newPool.step = 15; // activate step and set to unstable
                //newPool.trackId = tmpFrame(*r).clone(); // copy the traced img to trackId

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

float
traceObj::matComp(cv::Mat newId, cv::Mat srcId){
    return 1.; //Just return true by now, will do some more work future
}

void
traceObj::doCascadeDetect(){
    double scale = 1.1;
    Mat smallFrame(srcFrame->size(), CV_8UC1);
    cvtColor(*srcFrame, smallFrame, CV_BGR2GRAY);
//  resize(smallFrame, smallFrame, Size(256, 256));
    equalizeHist(smallFrame, smallFrame);

    cascade.detectMultiScale(smallFrame, goalObjects,scale,3,
                        CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_CANNY_PRUNING,
                        Size(100,100), Size(200,200) );
    pushPool(goalObjects);
}

void
traceObj::doBlobDetect(){
    Mat mask_frame(srcFrame->size(), CV_8UC1, Scalar(0));
    Mat tmp_frame = mask_frame;
    for(vector<Vec3b>::iterator c = keyColors.begin(); c != keyColors.end(); c++){
        colorKey(*srcFrame, tmp_frame, *c, hsvRange);
        cv::max(mask_frame, tmp_frame, mask_frame);
    }
    GaussianBlur (mask_frame, mask_frame, Size(51,51), 0, 0, BORDER_DEFAULT);
    threshold( mask_frame, mask_frame, 20, 255, CV_THRESH_BINARY);

    SimpleBlobDetector::Params param;
    param.filterByColor = false;
    param.filterByArea = true;
    param.filterByInertia = false;
    param.filterByConvexity = false;
    param.filterByCircularity = false;
    param.minArea = minArea;
    param.maxArea = maxArea;

    cv::SimpleBlobDetector blobSampler(param);
    blobSampler.create("SimpleBlob");
    vector<cv::KeyPoint> keypts;

    blobSampler.detect(mask_frame,keypts);
    goalObjects.clear();
    for(vector<cv::KeyPoint>::iterator r=keypts.begin();r!=keypts.end();r++){
        drawCross(*srcFrame, r->pt, cv::Scalar(0,255,255), 3);
        cv::circle(*srcFrame, r->pt, r->size, cv::Scalar(255,255,255) );
        if(r->size > 5)
            goalObjects.push_back(Rect(r->pt.x-r->size/2, r->pt.y-r->size/2, r->size, r->size));
    }
    pushPool(goalObjects);
}

void
traceObj::colorKey(cv::Mat& src, cv::Mat& dst, cv::Vec3b _keyColor, cv::Vec3b hsvRange){
    cv::Mat tmp_frame;
    dst = cv::Mat(src.size(), CV_8U, 255);
    //vector<cv::Mat> channels;

    cv::cvtColor(src, tmp_frame, CV_BGR2HSV);

    // Iterate through frame for calculation
    cv::MatIterator_<uchar> dst_it = dst.begin<uchar>();
    cv::MatIterator_<cv::Vec3b> it = tmp_frame.begin<cv::Vec3b>(),
                                it_end = tmp_frame.end<cv::Vec3b>();
    for(; it != it_end; ){
        cv::Vec3b& pixel = *it++; // reference to pixel in tmp_frame
        uchar& dst_pix = *dst_it++; // reference to mask frame pixel

        int tmp_hue = abs(pixel[0]-_keyColor[0]);
        tmp_hue = tmp_hue>90?(180-tmp_hue):tmp_hue; // Find the shortest distance in hue ring
        //dst_pix = 255 - linstep(0, hsvRange[0], tmp_hue);
        //dst_pix = std::min((int)dst_pix, 255 - linstep(0, hsvRange[1], abs(pixel[1]-_keyColor[1])));
        //dst_pix = std::min((int)dst_pix, 255 - linstep(0, hsvRange[2], abs(pixel[2]-_keyColor[2])));
        dst_pix = isInside(0, hsvRange[0], tmp_hue)*255;
    }
}

