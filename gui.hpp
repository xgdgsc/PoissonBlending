#ifndef GUI_H
#define GUI_H
#include"common_inc.h"
#include"data.h"
#include"poisson.h"

//highgui mouse handlers
bool drag=false;
Point2i mousePosSource,mousePosDest;

//mouse handler for source image display
void mouseHandlerSource(int event, int x, int y, int flags, void* userdata){

    if((event == CV_EVENT_LBUTTONDOWN )&& !drag){
        mousePosSource=Point2i(x,y);
        drag=true;
    }

    if((event == CV_EVENT_MOUSEMOVE)  &&drag){
        //imshow("Source Image",sourceImg);
        subImg=sourceImg.clone();
        rectangle(subImg,mousePosSource,Point(x,y),CV_RGB(0,255,0));
        imshow("Source Image",subImg);
    }

    if(event == CV_EVENT_LBUTTONUP && drag){
        subImg = sourceImg.clone();
        roiImg = Mat(subImg,Rect(mousePosSource.x,mousePosSource.y,x-mousePosSource.x,y-mousePosSource.y)).clone();
        namedWindow("Selected");
        imshow("Selected",roiImg);
        waitKey();
        destroyWindow("Selected");

        imshow("Source Image",sourceImg);
        drag=false;
    }

    if(event == CV_EVENT_RBUTTONUP){
        imshow("Source Image",sourceImg);
        drag=false;
    }
}

//mouse handler for destination image display
void mouseHandlerDest(int event, int x, int y, int flags, void* userdata){
    Mat roiDest;
    switch(event){
    case CV_EVENT_LBUTTONDOWN:
        mousePosDest = Point(x,y);
        roiDest=destImg.clone();
        rectangle(roiDest,mousePosDest,Point(mousePosDest.x+roiImg.cols,mousePosDest.y+roiImg.rows),CV_RGB(0,255,0));
        imshow("Dest Image",roiDest);
        break;
    case CV_EVENT_RBUTTONUP:
    case CV_EVENT_LBUTTONDBLCLK:
        if(mousePosDest.x+roiImg.cols>destImg.cols){
            roiImg=roiImg.colRange(0,destImg.cols-mousePosDest.x);
        }
        else if(mousePosDest.y+roiImg.rows>destImg.rows){
            roiImg=roiImg.rowRange(0,destImg.rows-mousePosDest.y);
        }

        blendedImg =blend(destImg,roiImg,mousePosDest);

        namedWindow("Blended Image");
        imshow("Blended Image",blendedImg);
        imwrite("blended.png",blendedImg);


        break;
    default:
        break;
    }
}

#endif // GUI_H
