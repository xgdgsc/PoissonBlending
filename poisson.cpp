
#include"poisson.h"
#define pi 3.14159265358979326

//calc X gradient
void calcGradX( const IplImage *img, IplImage *gx)
{
    int w = img->width;
    int h = img->height;
    int channel = img->nChannels;

    cvZero( gx );
    for(int i=0;i<h;i++)
        for(int j=0;j<w;j++)
            for(int c=0;c<channel;++c)
            {
                CV_IMAGE_ELEM(gx,float,i,j*channel+c) =
                    (float)CV_IMAGE_ELEM(img,uchar,i,(j+1)*channel+c) - (float)CV_IMAGE_ELEM(img,uchar,i,j*channel+c);
            }
}

//calc Y gradient
void calcGradY( const IplImage *img, IplImage *gy)
{
    int w = img->width;
    int h = img->height;
    int channel = img->nChannels;

    cvZero( gy );
    for(int i=0;i<h;i++)
        for(int j=0;j<w;j++)
            for(int c=0;c<channel;++c)
            {
                CV_IMAGE_ELEM(gy,float,i,j*channel+c) =
                    (float)CV_IMAGE_ELEM(img,uchar,(i+1),j*channel+c) - (float)CV_IMAGE_ELEM(img,uchar,i,j*channel+c);

            }
}

void dst(double *gtest, double *gfinal,int h,int w)
{
    unsigned long int idx;

    Mat temp = Mat(2*h+2,1,CV_32F);
    Mat res  = Mat(h,1,CV_32F);

    int p=0;
    for(int i=0;i<w;i++)
    {
        temp.at<float>(0,0) = 0.0;

        for(int j=0,r=1;j<h;j++,r++)
        {
            idx = j*w+i;
            temp.at<float>(r,0) = gtest[idx];
        }

        temp.at<float>(h+1,0)=0.0;

        for(int j=h-1, r=h+2;j>=0;j--,r++)
        {
            idx = j*w+i;
            temp.at<float>(r,0) = -1*gtest[idx];
        }

        Mat planes[] = {Mat_<float>(temp), Mat::zeros(temp.size(), CV_32F)};

        Mat complex1;
        merge(planes, 2, complex1);

        dft(complex1,complex1,0,0);

        Mat planes1[] = {Mat::zeros(complex1.size(), CV_32F), Mat::zeros(complex1.size(), CV_32F)};

        split(complex1, planes1);

        std::complex<double> two_i = std::sqrt(std::complex<double>(-1));

        double fac = -2*imag(two_i);

        for(int c=1,z=0;c<h+1;c++,z++)
        {
            res.at<float>(z,0) = planes1[1].at<float>(c,0)/fac;
        }

        for(int q=0,z=0;q<h;q++,z++)
        {
            idx = q*w+p;
            gfinal[idx] =  res.at<float>(z,0);
        }
        p++;
    }

}

void idst(double *gtest, double *gfinal,int h,int w)
{
    int nn = h+1;
    unsigned long int idx;
    dst(gtest,gfinal,h,w);
    for(int  i= 0;i<h;i++)
        for(int j=0;j<w;j++)
        {
            idx = i*w + j;
            gfinal[idx] = (double) (2*gfinal[idx])/nn;
        }


}

void transpose(double *mat, double *mat_t,int h,int w)
{

    Mat tmp = Mat(h,w,CV_32FC1);
    unsigned long int idx;
    for(int i = 0 ; i < h;i++)
    {
        for(int j = 0 ; j < w; j++)
        {

            idx = i*(w) + j;
            tmp.at<float>(i,j) = mat[idx];
        }
    }
    Mat tmp_t = tmp.t();

    for(int i = 0;i < tmp_t.size().height; i++)
        for(int j=0;j<tmp_t.size().width;j++)
        {
            idx = i*tmp_t.size().width + j;
            mat_t[idx] = tmp_t.at<float>(i,j);
        }

}

//calc lap of x direction
void lapx( const IplImage *img, IplImage *gxx)
{
    int w = img->width;
    int h = img->height;
    int channel = img->nChannels;

    cvZero( gxx );
    for(int i=0;i<h;i++)
        for(int j=0;j<w-1;j++)
            for(int c=0;c<channel;++c)
            {
                CV_IMAGE_ELEM(gxx,float,i,(j+1)*channel+c) =
                        (float)CV_IMAGE_ELEM(img,float,i,(j+1)*channel+c) - (float)CV_IMAGE_ELEM(img,float,i,j*channel+c);
            }
}

//calc lap of y direction
void lapy( const IplImage *img, IplImage *gyy)
{
    int w = img->width;
    int h = img->height;
    int channel = img->nChannels;

    cvZero( gyy );
    for(int i=0;i<h-1;i++)
        for(int j=0;j<w;j++)
            for(int c=0;c<channel;++c)
            {
                CV_IMAGE_ELEM(gyy,float,i+1,j*channel+c) =
                    (float)CV_IMAGE_ELEM(img,float,(i+1),j*channel+c) - (float)CV_IMAGE_ELEM(img,float,i,j*channel+c);

            }
}

//solve Poisson problem
void solvePoisson(const IplImage *img, IplImage *gxx , IplImage *gyy, Mat &result)
{

    int w = img->width;
    int h = img->height;

    unsigned long int idx,idx1;

    IplImage *lap  = cvCreateImage(cvGetSize(img), 32, 1);

    for(int i =0;i<h;i++)
        for(int j=0;j<w;j++)
            CV_IMAGE_ELEM(lap,float,i,j)=CV_IMAGE_ELEM(gyy,float,i,j)+CV_IMAGE_ELEM(gxx,float,i,j);

    Mat bound(img);

    for(int i =1;i<h-1;i++)
        for(int j=1;j<w-1;j++)
        {
            bound.at<uchar>(i,j) = 0.0;
        }

    double *f_bp = new double[h*w];


    for(int i =1;i<h-1;i++)
        for(int j=1;j<w-1;j++)
        {
            idx=i*w + j;
            f_bp[idx] = -4*(int)bound.at<uchar>(i,j) + (int)bound.at<uchar>(i,(j+1)) + (int)bound.at<uchar>(i,(j-1))
                    + (int)bound.at<uchar>(i-1,j) + (int)bound.at<uchar>(i+1,j);
        }


    Mat diff = Mat(h,w,CV_32FC1);
    for(int i =0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            idx = i*w+j;
            diff.at<float>(i,j) = (CV_IMAGE_ELEM(lap,float,i,j) - f_bp[idx]);
        }
    }
    double *gtest = new double[(h-2)*(w-2)];
    for(int i = 0 ; i < h-2;i++)
    {
        for(int j = 0 ; j < w-2; j++)
        {
            idx = i*(w-2) + j;
            gtest[idx] = diff.at<float>(i+1,j+1);

        }
    }

    double *gfinal = new double[(h-2)*(w-2)];
    double *gfinal_t = new double[(h-2)*(w-2)];
    double *denom = new double[(h-2)*(w-2)];
    double *f3 = new double[(h-2)*(w-2)];
    double *f3_t = new double[(h-2)*(w-2)];
    double *img_d = new double[(h)*(w)];

    dst(gtest,gfinal,h-2,w-2);

    transpose(gfinal,gfinal_t,h-2,w-2);

    dst(gfinal_t,gfinal,w-2,h-2);

    transpose(gfinal,gfinal_t,w-2,h-2);

    int cy=1;

    for(int i = 0 ; i < w-2;i++,cy++)
    {
        for(int j = 0,cx = 1; j < h-2; j++,cx++)
        {
            idx = j*(w-2) + i;
            denom[idx] = (float) 2*cos(pi*cy/( (double) (w-1))) - 2 + 2*cos(pi*cx/((double) (h-1))) - 2;

        }
    }

    for(idx = 0 ; idx <(unsigned int) (w-2)*(h-2) ;idx++)
    {
        gfinal_t[idx] = gfinal_t[idx]/denom[idx];
    }

    idst(gfinal_t,f3,h-2,w-2);

    transpose(f3,f3_t,h-2,w-2);

    idst(f3_t,f3,w-2,h-2);

    transpose(f3,f3_t,w-2,h-2);

    for(int i = 0 ; i < h;i++)
    {
        for(int j = 0 ; j < w; j++)
        {
            idx = i*w + j;
            img_d[idx] = (double)CV_IMAGE_ELEM(img,uchar,i,j);
        }
    }
    for(int i = 1 ; i < h-1;i++)
    {
        for(int j = 1 ; j < w-1; j++)
        {
            idx = i*w + j;
            img_d[idx] = 0.0;
        }
    }
    for(int i = 1,id1=0 ; i < h-1;i++,id1++)
    {
        for(int j = 1,id2=0 ; j < w-1; j++,id2++)
        {
            idx = i*w + j;
            idx1= id1*(w-2) + id2;
            img_d[idx] = f3_t[idx1];
        }
    }

    for(int i = 0 ; i < h;i++)
    {
        for(int j = 0 ; j < w; j++)
        {
            idx = i*w + j;
            if(img_d[idx] < 0.0)
                result.at<uchar>(i,j) = 0;
            else if(img_d[idx] > 255.0)
                result.at<uchar>(i,j) = 255.0;
            else
                result.at<uchar>(i,j) = img_d[idx];
        }
    }
}

//display blended image
//void showBlended(const char *name, IplImage *img)
//{
//    cvNamedWindow(name);
//    cvShowImage(name,img);
//}

//blending function
Mat blend(Mat dest, Mat roi,Point blendPos)
{
    //initialize data structure
    IplImage *destI = new IplImage(dest);
    IplImage *roiI = new IplImage(roi);
    int posX = blendPos.x;
    int posY = blendPos.y;

    IplImage *gradDestX  = cvCreateImage(cvGetSize(destI), 32, 3);
    IplImage *gradDestY  = cvCreateImage(cvGetSize(destI), 32, 3);

    IplImage *gradRoiX  = cvCreateImage(cvGetSize(roiI), 32, 3);
    IplImage *gradRoiY  = cvCreateImage(cvGetSize(roiI), 32, 3);

    IplImage *SI    = cvCreateImage(cvGetSize(destI), 8, 3);
    IplImage *erosionI  = cvCreateImage(cvGetSize(destI), 8, 3);
    IplImage *resultI  = cvCreateImage(cvGetSize(destI), 8, 3);

    //zero
    cvZero(SI);
    cvZero(resultI);

    int widthDest = destI->width;
    int heightDest = destI->height;
    int channelDest = destI->nChannels;

    int widthRoi = roiI->width;
    int heightRoi = roiI->height;

    //calc gradients
    calcGradX(destI,gradDestX);
    calcGradY(destI,gradDestY);

    calcGradX(roiI,gradRoiX);
    calcGradY(roiI,gradRoiY);

    //mask
    for(int i=posY, ii =0;i<posY+heightRoi;i++,ii++)
        for(int j=0,jj=posX;j<widthRoi;j++,jj++)
            for(int c=0;c<channelDest;++c)
            {
                CV_IMAGE_ELEM(SI,uchar,i,jj*channelDest+c) = 255;
            }

    //mask of x
    IplImage* bmaskx = cvCreateImage(cvGetSize(erosionI),32,3);
    cvConvertScale(SI,bmaskx,1.0/255.0,0.0);
    //mask of y
    IplImage* bmasky = cvCreateImage(cvGetSize(erosionI),32,3);
    cvConvertScale(SI,bmasky,1.0/255.0,0.0);

    for(int i=posY, ii =0;i<posY+heightRoi;i++,ii++)
        for(int j=0,jj=posX;j<widthRoi;j++,jj++)
            for(int c=0;c<channelDest;++c)
            {
                CV_IMAGE_ELEM(bmaskx,float,i,jj*channelDest+c) = CV_IMAGE_ELEM(gradRoiX,float,ii,j*channelDest+c);
                CV_IMAGE_ELEM(bmasky,float,i,jj*channelDest+c) = CV_IMAGE_ELEM(gradRoiY,float,ii,j*channelDest+c);
            }

    //erode si
    cvErode(SI,erosionI,NULL,1);

    IplImage* smask = cvCreateImage(cvGetSize(erosionI),32,3);
    cvConvertScale(erosionI,smask,1.0/255.0,0.0);

    IplImage* srx32 = cvCreateImage(cvGetSize(resultI),32,3);
    cvConvertScale(resultI,srx32,1.0/255.0,0.0);

    IplImage* sry32 = cvCreateImage(cvGetSize(resultI),32,3);
    cvConvertScale(resultI,sry32,1.0/255.0,0.0);

    for(int i=0;i < heightDest; i++)
        for(int j=0; j < widthDest; j++)
            for(int c=0;c<channelDest;++c)
            {
                CV_IMAGE_ELEM(srx32,float,i,j*channelDest+c) =
                    (CV_IMAGE_ELEM(bmaskx,float,i,j*channelDest+c)*CV_IMAGE_ELEM(smask,float,i,j*channelDest+c));
                CV_IMAGE_ELEM(sry32,float,i,j*channelDest+c) =
                    (CV_IMAGE_ELEM(bmasky,float,i,j*channelDest+c)*CV_IMAGE_ELEM(smask,float,i,j*channelDest+c));
            }

    //not operation
    cvNot(erosionI,erosionI);

    IplImage* smask1 = cvCreateImage(cvGetSize(erosionI),32,3);
    cvConvertScale(erosionI,smask1,1.0/255.0,0.0);

    IplImage* grx32 = cvCreateImage(cvGetSize(resultI),32,3);
    cvConvertScale(resultI,grx32,1.0/255.0,0.0);

    IplImage* gry32 = cvCreateImage(cvGetSize(resultI),32,3);
    cvConvertScale(resultI,gry32,1.0/255.0,0.0);

    for(int i=0;i < heightDest; i++)
        for(int j=0; j < widthDest; j++)
            for(int c=0;c<channelDest;++c)
            {
                CV_IMAGE_ELEM(grx32,float,i,j*channelDest+c) =
                    (CV_IMAGE_ELEM(gradDestX,float,i,j*channelDest+c)*CV_IMAGE_ELEM(smask1,float,i,j*channelDest+c));
                CV_IMAGE_ELEM(gry32,float,i,j*channelDest+c) =
                    (CV_IMAGE_ELEM(gradDestY,float,i,j*channelDest+c)*CV_IMAGE_ELEM(smask1,float,i,j*channelDest+c));
            }

    IplImage* fx = cvCreateImage(cvGetSize(resultI),32,3);
    IplImage* fy = cvCreateImage(cvGetSize(resultI),32,3);

    for(int i=0;i < heightDest; i++)
        for(int j=0; j < widthDest; j++)
            for(int c=0;c<channelDest;++c)
            {
                CV_IMAGE_ELEM(fx,float,i,j*channelDest+c) =
                    (CV_IMAGE_ELEM(grx32,float,i,j*channelDest+c)+CV_IMAGE_ELEM(srx32,float,i,j*channelDest+c));
                CV_IMAGE_ELEM(fy,float,i,j*channelDest+c) =
                    (CV_IMAGE_ELEM(gry32,float,i,j*channelDest+c)+CV_IMAGE_ELEM(sry32,float,i,j*channelDest+c));
            }

    IplImage *gxx  = cvCreateImage(cvGetSize(destI), 32, 3);
    IplImage *gyy  = cvCreateImage(cvGetSize(destI), 32, 3);

    //calc laps
    lapx(fx,gxx);
    lapy(fy,gyy);

    //create channels
    IplImage *rx_channel = cvCreateImage(cvGetSize(destI), IPL_DEPTH_32F, 1 );
    IplImage *gx_channel = cvCreateImage(cvGetSize(destI), IPL_DEPTH_32F, 1 );
    IplImage *bx_channel = cvCreateImage(cvGetSize(destI), IPL_DEPTH_32F, 1 );

    cvCvtPixToPlane(gxx, rx_channel, gx_channel, bx_channel,0);

    IplImage *ry_channel = cvCreateImage(cvGetSize(destI), IPL_DEPTH_32F, 1 );
    IplImage *gy_channel = cvCreateImage(cvGetSize(destI), IPL_DEPTH_32F, 1 );
    IplImage *by_channel = cvCreateImage(cvGetSize(destI), IPL_DEPTH_32F, 1 );

    cvCvtPixToPlane(gyy, ry_channel, gy_channel, by_channel,0);

    IplImage *r_channel = cvCreateImage(cvGetSize(destI), 8, 1 );
    IplImage *g_channel = cvCreateImage(cvGetSize(destI), 8, 1 );
    IplImage *b_channel = cvCreateImage(cvGetSize(destI), 8, 1 );

    cvCvtPixToPlane(destI, r_channel, g_channel, b_channel,0);

    Mat resultr = Mat(heightDest,widthDest,CV_8UC1);
    Mat resultg = Mat(heightDest,widthDest,CV_8UC1);
    Mat resultb = Mat(heightDest,widthDest,CV_8UC1);

    //solve poisson problem
    solvePoisson(r_channel,rx_channel, ry_channel,resultr);
    solvePoisson(g_channel,gx_channel, gy_channel,resultg);
    solvePoisson(b_channel,bx_channel, by_channel,resultb);

    IplImage *final = cvCreateImage(cvGetSize(destI), 8, 3 );

    for(int i=0;i<heightDest;i++)
        for(int j=0;j<widthDest;j++)
        {
            CV_IMAGE_ELEM(final,uchar,i,j*3+0) = resultr.at<uchar>(i,j);
            CV_IMAGE_ELEM(final,uchar,i,j*3+1) = resultg.at<uchar>(i,j);
            CV_IMAGE_ELEM(final,uchar,i,j*3+2) = resultb.at<uchar>(i,j);
        }

    return Mat(final);
}

