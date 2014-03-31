#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>  
#include <iostream>
#include <string.h>

using namespace std;
using namespace cv;

static CascadeClassifier cascade;
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

vector<Rect> objects;

void detectObj(Mat srcFrame);

string cascade_name = "haarcascade_frontalface_alt.xml";

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

	while (true){
		cap >> frame;
		if (frame.empty())
			continue;
		if (!objects.empty())
		for (vector<cv::Rect>::iterator r = objects.begin(); r != objects.end(); ++r)
			rectangle(frame, *r, colors[(r - objects.begin())%8], 1, 8, 1);
		imshow("windata",frame);
	}
	return 0;
}

void detectObj(Mat srcframe)
{
	double scale = 1.1;
	cvtColor(srcframe, srcframe, CV_BGR2GRAY);
	resize(srcframe, srcframe, Size(256, 256));
	equalizeHist(srcframe, srcframe);

	if (cascade.empty())
		return;

	double t = (double)cvGetTickCount(); // start evaluating process time
	cascade.detectMultiScale(srcframe, objects,scale,3, 
					0|CV_HAAR_SCALE_IMAGE,
					Size(50,50), Size(200,200) );
	t = (double)cvGetTickCount() - t;
	//printf("detection time = %gms/n", t / ((double)cvGetTickFrequency()*1000.));

	// Display detect time cost on screen overlay
	displayOverlay("windata", "detection time = " + to_string(t / ((double)cvGetTickFrequency()*1000.)) + "ms");

	return;
}