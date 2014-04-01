#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>

#include <stdio.h>  
#include <iostream>
#include <string.h>
#include <functional>
#include <chrono>
#include <future>
#include <cstdio>

using namespace std;
using namespace cv;

// Timer implemention
class later{
    public:
        template <class callable, class... arguments>  later(int after, bool async, callable&& f, arguments&&... args){
            function<typename result_of<callable(arguments...)>::type()> task(bind(forward<callable>(f), forward<arguments>(args)...));
            if (async){
                thread([after, task]() {
                    this_thread::sleep_for(std::chrono::milliseconds(after));
                    task();
                }).detach();
            }
            else{
                this_thread::sleep_for(std::chrono::milliseconds(after));
                task();
            }
        }
};

// setting for capcam global viarables
vector<Rect> objects;
static CascadeClassifier cascade;
static int sampleRate = 1000; // in ms
static CvScalar colors[] =
{
	{ { 0, 0, 255 } },
	{ { 0, 128, 255 } },
	{ { 0, 255, 255 } },
	{ { 0, 255, 0 } },
	{ { 255, 128, 0 } },
	{ { 255, 255, 0 } },
	{ { 255, 0, 0 } },
	{ { 255, 0, 255 } }
};
string cascade_name = "data/haarcascade_frontalface_alt.xml";

// Declare of functions
void detectObj(Mat& srcFrame);
void updateGraph();

int main(int argc, char** argv)
{
	//Initialize Cascade system
	cascade.load(cascade_name);
	if (cascade.empty())
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}

	//Initialize Camera/VideoInput
	VideoCapture cap;
	if (argc >1)
	{
		cap.open(argv[1]);
	}else
		cap.open(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	
	Mat frame;
	namedWindow("windata");

    // Start the detect timer
    later(sampleRate, true, &detectObj, frame);
    
	while (true){
		cap >> frame;
		if (frame.empty())
			continue;
		if (!objects.empty())
		for (vector<cv::Rect>::iterator r = objects.begin(); r != objects.end(); ++r)
			rectangle(frame, *r, colors[(r - objects.begin())%8], 1, 8, 1);
		imshow("windata",frame);
        int keyCode = waitKey(20);
        if (keyCode == 'q' || keyCode == 'Q')
            break;
	}
	return 0;
}

void detectObj(Mat& srcframe)
{
    
    if (srcframe.empty())
        return;
	
    double scale = 1.1;
    Mat smallFrame;
	cvtColor(srcframe, smallFrame, CV_BGR2GRAY);
	resize(smallFrame, smallFrame, Size(256, 256));
	equalizeHist(smallFrame, smallFrame);

	if (cascade.empty())
		return;

	double t = (double)cvGetTickCount(); // start evaluating process time
	cascade.detectMultiScale(smallFrame, objects,scale,3,
					0|CV_HAAR_SCALE_IMAGE,
					Size(50,50), Size(200,200) );
	t = (double)cvGetTickCount() - t;
	printf("detection time = %gms/n", t / ((double)cvGetTickFrequency()*1000.));

	// Display detect time cost on screen overlay
    //	displayOverlay("windata", "detection time = " + to_string(t / ((double)cvGetTickFrequency()*1000.)) + "ms");

	return;
}