#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>

#include <stdio.h>  
#include <iostream>
#include <string.h>
#include <chrono>
#include <future>

using namespace std;
using namespace cv;

// Timer implemention
class later{
    public:
        template <class callable, class... arguments>  later(int after, bool async, int loop, callable&& f, arguments&&... args){
            function<typename std::result_of<callable(arguments...)>::type()> task(bind(forward<callable>(f), forward<arguments>(args)...));
            if (async){
                thread([after, task, loop]() {
                    if (loop == 0) {
                        while (true) {
                            this_thread::sleep_for(std::chrono::milliseconds(after));
                            task();
                        }
                    }else{
                        for (int i = loop; i >= 0; --i) {
                            this_thread::sleep_for(std::chrono::milliseconds(after));
                            task();
                        }
                    }
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
static int sampleRate = 200; // in ms
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
Mat frame;
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
	
	namedWindow("windata");

    // Start the detect timer
    later(sampleRate, true, 0, &detectObj, frame);
    
	while (true){
		cap >> frame;
		if (frame.empty())
			continue;
        Mat dispFrame(frame.size(),CV_8UC3);
        dispFrame = frame;
		if (!objects.empty())
            for (vector<cv::Rect>::iterator r = objects.begin(); r != objects.end(); ++r)
                rectangle(dispFrame, *r, colors[(r - objects.begin())%8], 1, 8, 0);
		imshow("windata",dispFrame);
        int keyCode = waitKey(20);
        if (keyCode == 'q' || keyCode == 'Q')
            break;
	}
	return 0;
}

void detectObj(Mat& srcframe)
{
    
    if (frame.empty()){
        printf("No input frames\n");
        return;
    }
	
    double scale = 1.1;
    Mat smallFrame(cv::Size(640,480), CV_8UC1);
	cvtColor(frame, smallFrame, CV_BGR2GRAY);
//	resize(smallFrame, smallFrame, Size(256, 256));
	equalizeHist(smallFrame, smallFrame);

	if (cascade.empty())
		return;

	double t = (double)cvGetTickCount(); // start evaluating process time
	cascade.detectMultiScale(smallFrame, objects,scale,3,
					CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_CANNY_PRUNING,
					Size(50,50), Size(200,200) );
	t = (double)cvGetTickCount() - t;
    printf("detection time = %gms\n", t / ((double)cvGetTickFrequency()*1000.));

	return;
}