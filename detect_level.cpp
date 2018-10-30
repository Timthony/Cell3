//
// Created by arcstone_mems on 2018/10/30.
//

#include "detect_level.h"

Point detect_level::MoveDetect(Mat temp, Mat frame, int img_h, int img_w,int img_k)
{
    Mat result = frame.clone();
    temp = temp(Rect(0.25*img_w,0.45*img_h,img_w*0.6, img_h*0.12));
    frame = frame(Rect(0.25*img_w,0.45*img_h,img_w*0.6, img_h*0.12));
    //1.将background和frame转为灰度图
    Mat gray1,gray2;
    cvtColor(temp, gray1, CV_BGR2GRAY);
    cvtColor(frame, gray2, CV_BGR2GRAY);
    //2.将background和frame做差
    Mat diff;
    absdiff(gray1, gray2, diff);
    //imshow("diff", diff);
    //3.对差值图diff_thresh进行阈值化处理
    Mat diff_thresh;
    threshold(diff, diff_thresh, 50, 155, CV_THRESH_BINARY);
    //imshow("diff_thresh", diff_thresh);

    //4.腐蚀
    Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3,3));
    Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(18,18));
    //erode(diff_thresh, diff_thresh, kernel_erode);
    //imshow("erode", diff_thresh);

    //5.膨胀
    dilate(diff_thresh, diff_thresh, kernel_dilate);
    //imshow("dilate", diff_thresh);

//    //6.查找并绘制轮廓
    vector<vector<Point>> contours;
    findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//    vector<Point> maxcontours = contours[0];
//    for (int j = 0; j < contours.size(); j++)
//    {
//
//        if(maxcontours.size() < contours[j].size())
//        {
//            maxcontours = contours[j];
//        }
//    }
//
//    //contours.cvseq
    Point offset(0.25*img_w,0.45*img_h);//增加轮廓的偏置
    drawContours(result, contours, -1, Scalar(0,0,255), 1, 8, noArray(), INT_MAX, offset);

//    //7.查找正外接矩形
    vector<Rect> boundRect(contours.size());
    Rect maxboundRect(0,0,1,1);
//    Rect boundRect1;
//    boundRect1 = boundingRect(maxcontours);
//    rectangle(result, boundRect1, Scalar(0,255,0),2);
//
    for (int i = 0; i < contours.size(); i++)
    {
        boundRect[i] = boundingRect(contours[i]);
        if(maxboundRect.area() < boundRect[i].area())
        {
            maxboundRect = boundRect[i];
        }
    }
    //定义外接矩形最右边的位置
    int x_right;
    int y_right;
    Point p_cur;
    x_right = maxboundRect.x + maxboundRect.width + 0.25*img_w;
    if(x_right > 200 && x_right < 600 && img_k>243 && img_k<325)  //当播放的帧数与先前观测的相同时。
    {
        p_cur.x = x_right;
        p_cur.y = 0.45*img_h + 0.06*img_h;
        line(result, Point(x_right-10, 200), Point(x_right-10,400), Scalar(0,255,0),2,CV_AA);
    }
    //cout<<"液面的中心点为："<<p_cur<<endl;       //当前页面的位置
    //imshow("window", result);
    return p_cur;
}