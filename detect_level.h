//
// Created by arcstone_mems on 2018/10/30.
// 检测液面的位置，返回液面位置的坐标，也就是页面位置标志点P的横坐标，纵坐标
//

#ifndef CELL3_DETECT_LEVEL_H
#define CELL3_DETECT_LEVEL_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;


class detect_level {
private:
    int img_h;
    int img_w;
    int img_k;//记录当前的帧数
    Mat temp;
    Mat frame;
public:
    Point MoveDetect(Mat temp, Mat frame, int img_h, int img_w,int img_k);


};


#endif //CELL3_DETECT_LEVEL_H
