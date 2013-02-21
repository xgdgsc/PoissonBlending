
#include"gui.hpp"
#include"data.h"
using namespace std;
using namespace cv;

//argc = 3
int main(int argc,char** argv)
{
    if(argc<3){
        cout<<"Usage:"<<argv[0]<<" source_image destination_image"<<endl;
        exit(1);
    }

    //load images
    try{
        sourceImg=imread(argv[1]);
        destImg=imread(argv[2]);
    }
    catch(Exception e){
        cout<<e.what()<<endl;
    }

    //display source image
    namedWindow("Source Image");
    setMouseCallback("Source Image",mouseHandlerSource);
    imshow("Source Image",sourceImg);

    //display destination image
    namedWindow(("Dest Image"));
    setMouseCallback("Dest Image",mouseHandlerDest);
    imshow("Dest Image",destImg);

    waitKey();

    return 0;
}

