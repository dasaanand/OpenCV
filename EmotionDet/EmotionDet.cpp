
#include <OpenCV/OpenCV.h>
#include <cassert>
#include <iostream>


const char  * WINDOW_NAME  = "Emotion Recognition";
const CFIndex CASCADE_NAME_LEN = 2048;
char CASCADE_NAME[10][CASCADE_NAME_LEN] = {"haarcascade_frontalface_alt_tree.xml","haarcascade_mcs_mouth.xml", "haarcascade_eye_new.xml", "haarcascade_mcs_nose.xml", "haarcascade_anger.xml", "haarcascade_disgust.xml", "haarcascade_fear.xml", "haarcascade_sad.xml", "haarcascade_surprise.xml", "haarcascade_smile.xml"} ;
clock_t t1, t2;


using namespace std;

string emotion[] = {"Anger","Disgust","Fear","Sad","Surprise","Smile"};

int Resize(int n, int m, int l)             //Gets a number between the range specified
{
    if(n<=l && l<=m)
		return l;
    else if(l<n)
		return Resize(n, m, l*2);		//if l<n then double the value
    else
		return Resize(n, m, l/3);		//if not l = l/3;
    t2 = clock();
}

int Random(int n, int m)                   //Generates the random number using time.
{
    t1 = clock();
	unsigned int x = x*x*(t2-t1)*10;
    
	int no = Resize(n, m, (1+x%10));	//reducing the random no within the given range
	return no;
}

int main (int argc, char * const argv[]) 
{
   	string imgEmotion;
    const int scale = 2;
	
    CFBundleRef mainBundle  = CFBundleGetMainBundle ();
    assert (mainBundle);
	CFURLRef    cascade_url[10];
	
	cascade_url[0] = CFBundleCopyResourceURL (mainBundle, CFSTR("haarcascade_frontalface_alt_tree"), CFSTR("xml"), NULL);
    cascade_url[1] = CFBundleCopyResourceURL (mainBundle, CFSTR("haarcascade_mcs_mouth"), CFSTR("xml"), NULL);
    cascade_url[2] = CFBundleCopyResourceURL (mainBundle, CFSTR("haarcascade_eye_new"), CFSTR("xml"), NULL);
    cascade_url[3] = CFBundleCopyResourceURL (mainBundle, CFSTR("haarcascade_mcs_nose"), CFSTR("xml"), NULL);
    cascade_url[4] = CFBundleCopyResourceURL (mainBundle, CFSTR("haarcascade_anger"), CFSTR("xml"), NULL);
    cascade_url[5] = CFBundleCopyResourceURL (mainBundle, CFSTR("haarcascade_disgust"), CFSTR("xml"), NULL);
    cascade_url[6] = CFBundleCopyResourceURL (mainBundle, CFSTR("haarcascade_fear"), CFSTR("xml"), NULL);
    cascade_url[7] = CFBundleCopyResourceURL (mainBundle, CFSTR("haarcascade_sad"), CFSTR("xml"), NULL);
    cascade_url[8] = CFBundleCopyResourceURL (mainBundle, CFSTR("haarcascade_surprise"), CFSTR("xml"), NULL);
    cascade_url[9] = CFBundleCopyResourceURL (mainBundle, CFSTR("haarcascade_smile"), CFSTR("xml"), NULL);	
	int i;
	Boolean got_it;
	
	for (i=0; i<10; i++) {
		assert (cascade_url[i]);
		got_it = CFURLGetFileSystemRepresentation (cascade_url[i], true, reinterpret_cast<UInt8 *>(CASCADE_NAME[i]), CASCADE_NAME_LEN);
		if (! got_it)
			abort ();
	}
	
	
    cvNamedWindow (WINDOW_NAME, CV_WINDOW_AUTOSIZE);
	CvCapture * camera = cvCreateCameraCapture (CV_CAP_ANY);
    CvHaarClassifierCascade* cascade[10];
	CvMemStorage* storage[10];	
	
	for (i=0; i<10; i++) {
		cascade[i] = (CvHaarClassifierCascade*) cvLoad (CASCADE_NAME[i], 0, 0, 0);
		storage[i] = cvCreateMemStorage(0);
		assert(storage[i]);
	}
	
    if (! camera)
        abort ();
	
    IplImage * current_frame = cvQueryFrame (camera);
    IplImage * draw_image = cvCreateImage(cvSize (current_frame->width, current_frame->height), IPL_DEPTH_8U, 3);
    IplImage * gray_image = cvCreateImage(cvSize (current_frame->width, current_frame->height), IPL_DEPTH_8U, 1);
    IplImage * small_image = cvCreateImage(cvSize (current_frame->width / scale, current_frame->height / scale), IPL_DEPTH_8U, 1);
    assert (current_frame && gray_image && draw_image);
	
	
    while (current_frame = cvQueryFrame (camera))
    {
		
        cvCvtColor (current_frame, gray_image, CV_BGR2GRAY);
        cvResize (gray_image, small_image, CV_INTER_LINEAR);
        
        CvSeq* faces = cvHaarDetectObjects (small_image, cascade[0], storage[0], 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize (22, 22));
        CvSeq* mouth = cvHaarDetectObjects(small_image, cascade[9], storage[9], 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(25, 15));
        CvSeq* eye = cvHaarDetectObjects(small_image, cascade[2], storage[2], 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(22, 22));
		CvSeq* nose= cvHaarDetectObjects(small_image, cascade[3], storage[3], 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(20, 17));
		CvSeq* smile= cvHaarDetectObjects(small_image, cascade[4], storage[4], 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(43, 19));		
		
        cvFlip (current_frame, draw_image, 1);
		CvRect* r[5];
		CvPoint center[5];
		int radius[5];
		
        for (int i = 0; i < (faces ? faces->total : 0); i++)
        {
            r[0] = (CvRect*) cvGetSeqElem (faces, i);
            center[0].x = cvRound((small_image->width - r[0]->width*0.5 - r[0]->x) *scale);
            center[0].y = cvRound((r[0]->y + r[0]->height*0.5)*scale);
            radius[0] = cvRound((r[0]->width + r[0]->height)*0.25*scale);
			cvRectangle(draw_image, cvPoint((center[0].x)+radius[0], center[0].y+radius[0]), cvPoint(center[0].x-radius[0], center[0].y-radius[0]), CV_RGB(255,255,255), 2, 8, 0);
			
			
			r[1] = (CvRect*) cvGetSeqElem (mouth, i);
            center[1].x = cvRound((small_image->width - r[1]->width*0.5 - r[1]->x) *scale);
            center[1].y = cvRound((r[1]->y + r[1]->height*0.5)*scale);
            radius[1] = cvRound((r[1]->width + r[1]->height)*0.25*scale);
			cvRectangle(draw_image, cvPoint((center[1].x)+radius[1], center[1].y+radius[1]), cvPoint(center[1].x-radius[1], center[1].y-radius[1]), CV_RGB(255,0,0), 2, 8, 0);
			
			for (int j = 0; j < (eye ? eye->total : 0); j++)
			{
				r[2] = (CvRect*) cvGetSeqElem (eye, j);
				center[2].x = cvRound((small_image->width - r[2]->width*0.5 - r[2]->x) *scale);
				center[2].y = cvRound((r[2]->y + r[2]->height*0.5)*scale);
				radius[2] = cvRound((r[2]->width + r[2]->height)*0.25*scale);
				cvRectangle(draw_image, cvPoint((center[2].x)+radius[2], center[2].y+radius[2]), cvPoint(center[2].x-radius[2], center[2].y-radius[2]), CV_RGB(255,0,255), 2, 8, 0);
			}
			
			r[3] = (CvRect*) cvGetSeqElem (nose, i);
            center[3].x = cvRound((small_image->width - r[3]->width*0.5 - r[3]->x) *scale);
            center[3].y = cvRound((r[3]->y + r[3]->height*0.5)*scale);
            radius[3] = cvRound((r[3]->width + r[3]->height)*0.25*scale);
			cvRectangle(draw_image, cvPoint((center[3].x)+radius[3], center[3].y+radius[3]), cvPoint(center[3].x-radius[3], center[3].y-radius[3]), CV_RGB(0,255,255), 2, 8, 0);
			
        }
        
        cvShowImage (WINDOW_NAME, draw_image);
 		//cout << "\n-----" << emotion[Random(0,5)] << "-----";
        
    }
    
    return 0;
}