#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "detect_level.h"
using namespace std;
using namespace cv;

//-----------------------【全局变量声明】------------------------
Mat firstImage;                                  //保存第一张图像
int image_cols;
int image_rows;
int delay = 30;                                  //设置等待时间
int cur_frame_num = 0;                           //当前播放到的帧数
bool selectObject = false;                       //定义是否激活鼠标划取区域这个功能
Rect selection;                                  //定义一个矩形区域，表示鼠标划取的区域
Point origin;                                    //定义鼠标划取区域的起始点
Mat image;                                       //存储当前帧的图像
int trackObject = 0;
ofstream outfile;
string window_name = "flow tracking";            //定义窗口的名称
Mat result;                                      //定义最后输出的图像
vector<Point2f> points1;                         //存放需要跟踪的点
vector<Point2f> points1_cur;                     //存放原始点在当前帧的位置
Mat flow;                                        //光流矩阵
Mat cflow;
Mat gray;
Mat gray_prev;
vector<Point2f> points_temp;                     //存放需要跟踪的临时点
bool node_flag = false;                          //标识是否应该更新原始点的坐标
Mat frame;
Point level_cur;                                 //存储当前液面的位置
int num_origin;                                  //定义初始矩形框内点的个数
int num_end;                                     //定义最后时刻矩形框内还有多少原始跟踪点
int ks = 45;                                     //每隔多少帧暂停一次
int chou_begin = 273;                            //从多少帧开始抽
int chou_end = 319;                              //从多少帧结束抽
vector<Point2f> points_end;                      //最后时刻，框内的点
vector<Point2f> points_begin;                    //初始时刻，框内的点
Point level_end;

//-----------------------【全局函数声明】------------------------
float cal_density(int num_points);
float cal_shape(vector<Point2f> p_begin, vector<Point2f> p_end);
//--------------------------------------------------------------------
/*鼠标回调函数*/
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = abs(x - origin.x);
        selection.height = abs(y - origin.y);

        selection &= Rect(0, 0, image.cols, image.rows);     //保证selection在画面的里边
    }
    switch (event)
    {
        case EVENT_LBUTTONDOWN:
            origin = Point(x, y);
            selection = Rect(x, y, 0, 0);
            selectObject = true;
            break;
        case EVENT_LBUTTONUP:
            selectObject = false;
            if (selection.width > 0 && selection.height > 0)
            {
                trackObject = -1;
                outfile<<"这次取点为："<<endl;
            }
            break;
    }
}

void tracking_it(Mat &frame, Mat &output)
{
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    frame.copyTo(output);
    //在针管内勾选初始的长方形区域
    if(selectObject)
    {
        rectangle(output, Point(selection.x, selection.y),
                  Point(selection.x + selection.width,
                        selection.y + selection.height), Scalar(255, 0, 0), 0.5, 8);
    }
    //鼠标抬起，开始检测
    if(trackObject == -1)
    {
        //画出鼠标勾选的矩形
        rectangle(output, Point(selection.x, selection.y),
                  Point(selection.x + selection.width,
                        selection.y + selection.height), Scalar(225,255,0), 2, 8);
        cout<<"矩形的长度和高度为："<<selection.width<<","<<selection.height<<endl;
        //画出液面的位置以及计算的矩形框的范围
        line(output, Point(level_cur.x-10, 200), Point(level_cur.x-10,400), Scalar(0,255,0),2,CV_AA);
        rectangle(output, Point(level_cur.x-10-selection.width, selection.y),
                  Point(level_cur.x-10, selection.y + selection.height), Scalar(0, 255, 255), 2, 8);
        // 对整个画面的每个像素点进行遍历，如果此点在所画矩形内，那么加入点的集合
        if(!node_flag)
        {
            //只执行一次
            for (int y = 0; y < image_rows; y++)
            {
                for (int x = 0; x < image_cols; x++)
                {
                    Point p(x, y);
                    if(selection.contains(p))
                    {
                        points_temp.push_back(p);
                    }
                }
            }
            points1_cur = points_temp;
            cout<<"本次计算的点有"<<points_temp.size()<<"个"<<endl;
            num_origin = points_temp.size();
            points_begin = points_temp;
            node_flag = true;
            points_temp.clear();
        }
        //points1 = points_temp;//初始化的点集合
        if(gray_prev.empty())
        {
            gray.copyTo(gray_prev);
        }
        if(gray_prev.data)
        {
            calcOpticalFlowFarneback(gray_prev, gray, flow, 0.5, 2, 15, 3, 5, 1.2, 0);
            cvtColor(gray_prev, cflow, CV_GRAY2BGR);
            // 对选定的初始点进行标记，每播放一帧它们的坐标更新为新的，
            for (int i = 0; i < points1_cur.size(); i++)
            {
                const Point2f& fxy = flow.at<Point2f>(points1_cur[i].y, points1_cur[i].x);
                points1_cur[i].x = points1_cur[i].x + fxy.x;
                points1_cur[i].y = points1_cur[i].y + fxy.y;
                circle(output, points1_cur[i], 1, CV_RGB(255,0,0),-1);
            }
            swap(gray_prev, gray);
            //计算原始点在当前矩形框的数目，并且将这些点存入新的集合
            Rect selection_cur;                                //定义一个实时显示的矩形区域
            selection_cur.x = level_cur.x-10-selection.width;
            selection_cur.y = selection.y;
            selection_cur.width = selection.width;
            selection_cur.height = selection.height;
            vector<Point2f> p_cur_rect_tmp;
            vector<Point2f> p_cur_rect;//存放在当前帧矩形框跟踪点的集合
            //判断当前帧原始的跟踪点是否在该矩形区域，如果在，则返回新的集合
            for (int m = 0; m < points1_cur.size(); m++)
            {
                if(selection_cur.contains(points1_cur[m]))
                {
                    p_cur_rect_tmp.push_back(points1_cur[m]);
                }
            }
            p_cur_rect = p_cur_rect_tmp;
            p_cur_rect_tmp.clear();
            for (int j = 0; j < p_cur_rect.size(); j++)
            {
                circle(output, p_cur_rect[j], 1, CV_RGB(0,255,255),-1);
            }
            cout<<"当前帧中矩形框内的原始跟踪点还剩："<<p_cur_rect.size()<<"个"<<endl;
            //最后时刻的参数
            if(cur_frame_num == (chou_begin+ks))
            {
                level_end.x = level_cur.x-10;
                level_end.y = level_cur.y;
                num_end = p_cur_rect.size();
                points_end = p_cur_rect;
//                for (int z = 0; z < points_end.size(); z++)
//                {
//                    cout<<"最后时刻点的坐标为：("<<points_end[z].x<<","<<points_end[z].y<<")"<<endl;
//                }
                float density_begin = cal_density(num_origin);
                cout<<"初始密度为："<<density_begin<<endl;
                float density_end = cal_density(num_end);
                cout<<"最后密度为："<<density_end<<endl;
                float yingbian = cal_shape(points_begin, points_end);
                cout<<"最后应变为："<<yingbian<<endl;
            }
        }
    }
    imshow(window_name, output);
}
//计算密度的函数
float cal_density(int num_points)
{
    //就是在矩形区域内点的密度，num_points/S
    float p_density = float(num_points)/float(selection.width*selection.height);
    return p_density;
}
//计算应变的函数,待修改
float cal_shape(vector<Point2f> p_begin, vector<Point2f> p_end)
{
    //计算应变，正应变，到矩形框右侧的距离的均值
    float l_begin;//初始的间距值
    float l_sum = 0,l_avg;
    for (int i = 0; i < p_begin.size(); i++)
    {
        float l_tmp = abs(selection.x+selection.width - p_begin[i].x);
        l_sum = l_sum + l_tmp;
    }
    l_avg = l_sum/float(num_origin);
    float l_sum_end = 0;
    float l_avg_end;
    for (int j = 0; j < p_end.size(); j++)
    {
        float l_tmp_end = abs(level_end.x - p_end[j].x);
        l_sum_end = l_sum_end + l_tmp_end;
    }
    l_avg_end = l_sum_end/float(num_end);
    float yingbian = (l_avg_end-l_avg)/float(selection.width);
    return yingbian;
}


int main() {
    outfile.open("outdata.txt");
//------------------------------【更换源视频必调参数】-----------------
    VideoCapture capture("/Users/arcstone_mems_108/Desktop/keyan/githubproject/cell3/cmake-build-debug/test_4_6.avi");

//------------------------------------------------------------------
    capture>>firstImage;
    image_cols = firstImage.cols;
    image_rows = firstImage.rows;
    if(!capture.isOpened())
    {
        cout<<"原始视频未能正确打开！"<<endl;
    }
    Mat temp = firstImage;
    while(true)
    {
        capture>>frame;
        cur_frame_num++;
        cout<<"当前为第"<<cur_frame_num<<"帧"<<endl;
        outfile<<"当前为第"<<cur_frame_num<<"帧"<<endl;
        if(frame.empty())
        {
            break;
        }
        frame.copyTo(image);
        setMouseCallback(window_name, onMouse, 0);
        detect_level det;

        level_cur = det.MoveDetect(temp, frame, image_rows, image_cols,cur_frame_num);
        cout<<"当前液面的位置为："<<level_cur<<endl;
        if(!frame.empty())
        {
            tracking_it(image, result);//进行光流检测
        }
        if( cur_frame_num>=chou_begin && (cur_frame_num-chou_begin)%ks == 0
            || (cur_frame_num-chou_end) == 0)
        //从多少帧开始取点，并且隔多少帧再次取点，***换视频必调参数***
        {
            waitKey(0);
        }
        //等待，等待delay的ms如果没有操作，那么返回-1，视频将继续播放。按一下键，视频播放40帧
        if(delay>=0 && waitKey(delay)>=0)
        {
            waitKey(0);
        }
        temp = frame.clone();
        if(cur_frame_num == chou_begin + ks + 1)
        {
            float density_begin = cal_density(num_origin);
            cout<<"初始密度为："<<density_begin<<endl;
            float density_end = cal_density(num_end);
            cout<<"最后密度为："<<density_end<<endl;
            float yingbian = cal_shape(points_begin, points_end);
            cout<<"最后应变为："<<yingbian<<endl;
        }
    }
    return 0;

}

