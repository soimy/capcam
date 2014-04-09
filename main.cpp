#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "traceObj.h"
#include "later.h"

#include <stdio.h>  
#include <iostream>
#include <string.h>
#include <chrono>
#include <future>

using namespace std;
using namespace cv;


const string cascade_name = "data/haarcascade_frontalface_alt.xml";
int capWidth = 640;
int capHeight = 480;
// Declare of functions
//void detectObj(Mat& srcFrame);
//void updateGraph();

int main(int argc, char** argv)
{
	//Initialize Object tracing` system
    traceObj fishTrace;
    fishTrace.cascade_name = cascade_name;
    fishTrace.drawMat = true;
    fishTrace.sampleRate = 200;
    
	//Initialize Camera/VideoInput
	VideoCapture cap;
    Mat frame(Size(capWidth,capHeight),CV_8UC3);
	if (argc >1)
	{
		cap.open(argv[1]);
	}else
		cap.open(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, capWidth);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, capHeight);
    
	namedWindow("windata");

    // init and start traceObj
    if(!fishTrace.init())
        return -1;
    fishTrace.attachFrame(frame);
    
    // Start the video capture main loop
	while (true){
		cap >> frame;
		if (frame.empty())
			continue;
        Mat dispFrame(frame.size(),CV_8UC3);
        dispFrame = frame + fishTrace.overLay ;
//        for(vector<Rect>::iterator r = fishTrace.objects.begin(); r != fishTrace.objects.end(); r++){
//            rectangle(dispFrame, *r, fishTrace.colors[r-fishTrace.objects.begin()]);
//        }
		imshow("windata",dispFrame);
        fishTrace.update();
        int keyCode = waitKey(20);
        if (keyCode == 'q' || keyCode == 'Q'){
            fishTrace.stop();
            break;
        }
	}
	return 0;
}

