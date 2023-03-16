
/*     ____  ____________    __ ____   __     _   __________________  _  __
 __ / / / / / __/_  __/___/ // / /  / / ___| | / /  _/ __/  _/ __ \/ |/ /
/ // / /_/ /\ \  / / /___/ _  / /__/ /_/___/ |/ // /_\ \_/ // /_/ /    /
\___/\____/___/ /_/     /_//_/____/____/   |___/___/___/___/\____/_/|_/
*/
/**
 * @file        Buff detector
 * @details     Windows-10 + Qt Creator-5.14.2 + OpenCV-4.1.1
 * @author      zhaoqicheng zhaoqicheng2023@163.com
 * @version     1.0
 * @date        2023/3/15
 */
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
//using namespace cv;

/*参数声明--------------------------------------------------------------------------------------------*/
const string video_path = "F:\\hll-code\\video\\1.mp4";  // 设置视频文件路径
//cv::Mat element1 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));    // 设置内核1
//cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(25,25));  // 设置内核2
/*函数声明--------------------------------------------------------------------------------------------*/
//void ImageReprocessing(cv::Mat video_frame,vector<cv::Mat> channels,cv::Mat element1,cv::Mat element2);  // 图像预处理
//void IdentifyBUFF(cv::Mat video_frame,cv::Mat frame_plot,
//                  vector<vector<cv::Point> > contours,vector<cv::Vec4i> hierarchy,
//                  int area[],vector<cv::Point2d> points) ;  // 识别 BUFF
int LeastSquaresCircleFitting(vector<cv::Point2d> &m_Points, cv::Point2d &Centroid, double &dRadius);  // 拟合函数
/*主函数----------------------------------------------------------------------------------------------*/
int main() {
    cv::VideoCapture video(video_path);
    if (!video.isOpened()) {
        cout << "[WARN] Video open faild" << endl;
    }
    cv::Mat video_frame;

    while (true)
        {
            video >> video_frame;
            if (video_frame.empty())
                break;
            cv::Mat video_plot = video_frame.clone();

            vector<cv::Mat> channels;
            cv::split(video_frame, channels);  //分离通道
            cv::threshold(channels.at(2) - channels.at(0), video_frame, 100, 255, cv::THRESH_BINARY);//二值化
            cv::Mat struct1 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(9,9));
            cv::dilate(video_frame,video_frame,struct1);
            cv::Mat struct2 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(9,9));
            cv::erode(video_frame,video_frame,struct2);
            cv::namedWindow("test1", cv::WINDOW_NORMAL);
            cv::imshow("test1", video_frame);

            // cv::Mat element1 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5, 5));//设置内核1
            // cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));//设置内核2
            // cv::morphologyEx(video_frame, video_frame, cv::MORPH_CLOSE, element1);//开运算
            // cv::floodFill(video_frame, cv::Point(0, 0), cv::Scalar(0));//漫水
            // cv::namedWindow("test2", cv::WINDOW_NORMAL);
            // cv::imshow("test2", video_frame);
            // cv::morphologyEx(video_frame, video_frame, cv::MORPH_CLOSE, element2);//闭运算
            // cv::namedWindow("test3", cv::WINDOW_NORMAL);
            // cv::imshow("test3", video_frame);



            vector<vector<cv::Point> >contours;  //轮廓数组
            vector<cv::Vec4i>hierarchy; //一个参数
            cv::findContours(video_frame, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));  //提取所有轮廓并建立网状轮廓结构
            cv::Point2i center; //用来存放找到的目标的中心坐标

            int contour[20] = {0};
            for (int i = 0; i < int(contours.size()); i++) {//遍历检测的所有轮廓
                if (hierarchy[i][3] != -1) {  //有内嵌轮廓，说明是一个父轮廓
                    ++contour[hierarchy[i][3]]; //对该父轮廓进行记录
                }
            }

            for (int j = 0; j < int(contours.size()); j++) {//再次遍历所有轮廓
                if (contour[j] == 1) {//如果某轮廓对应数组的值为1，说明只要一个内嵌轮廓
                    int num = hierarchy[j][2]; //记录该轮廓的内嵌轮廓
                    cv::RotatedRect box = cv::minAreaRect(contours[num]); //包含该轮廓所有点
                    cv::Point2f vertex[4];
                    box.points(vertex);//将左下角，左上角，右上角，右下角存入点集
                    for (int i = 0; i < 4; i++) {
                        cv::line(video_plot, vertex[i], vertex[(i + 1) % 4], cv::Scalar(255, 0, 0), 4, cv::LINE_AA); //画线
                    }
                    center = (vertex[0] + vertex[2]) / 2; //返回中心坐标
                    cv::circle(video_plot,center,3,cv::Scalar(0, 255, 0),-1);
                    cv::putText(video_plot, "target", vertex[0], cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255, 255, 0),2.5);//打印字体
                }
            }
            cv::namedWindow("test2", cv::WINDOW_AUTOSIZE);
            cv::imshow("test2", video_plot);

            cv::namedWindow("BUFF", cv::WINDOW_NORMAL);
            cv::imshow("BUFF", video_plot);
            cv::waitKey(1);
        }


    return 0;
}

/*函数定义---------------------------------------------------------------------------------------------*/
/**
 * @brief 图像预处理
 * @param video_frame 图像
 * @param channels    分离后的通道
 * @param element1    内核1
 * @param element2    内核2
 */
void ImageReprocessing(cv::Mat video_frame,vector<cv::Mat> channels,cv::Mat element1,cv::Mat element2) {
    cv::split(video_frame,channels);  // 分离通道
    cv::threshold(channels.at(2) - channels.at(0),video_frame,100,255,cv::THRESH_BINARY_INV);  // 二值化图像
    cv::morphologyEx(video_frame,video_frame,cv::MORPH_OPEN,element1);   // 开运算
    // 能不能在这里写一句闭运算将外侧装甲板轮廓闭合？
    //    cv::namedWindow("test1",cv::WINDOW_NORMAL);
    //    cv::imshow("test1",video_frame);
    cv::floodFill(video_frame,cv::Point(0,0),cv::Scalar(0));  // 漫水法
    //    cv::namedWindow("test2",cv::WINDOW_NORMAL);
    //    cv::imshow("test2",video_frame);
    cv::morphologyEx(video_frame,video_frame,cv::MORPH_CLOSE,element2);  // 闭运算
    //    cv::namedWindow("test3",cv::WINDOW_NORMAL);
    //    cv::imshow("test3",video_frame);
}

/**
 * @brief IdentifyBUFF
 * @param video_frame
 * @param frame_plot
 * @param contours
 * @param hierarchy
 * @param area
 * @param points
 */
void IdentifyBUFF(cv::Mat video_frame,cv::Mat frame_plot,
                  vector<vector<cv::Point> > contours,vector<cv::Vec4i> hierarchy,
                  int area[],vector<cv::Point2d> points) {
    cv::findContours(video_frame, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);  //找轮廓
    for (int i = 0; i != int(hierarchy.size()); ++i) {
        area[i] = cv::contourArea(contours[i]);  //计算轮廓面积
        if (area[i] < 1000 ) {
            cv::Point2f rect[4];  // 用于保存装甲板最小外接矩阵的四个顶点坐标
            cv::RotatedRect box1 = minAreaRect(cv::Mat(contours[i])); //获取最小外接矩阵
            circle(frame_plot,cv::Point(box1.center.x, box1.center.y), 5, cv::Scalar(255, 0, 0), -1, 8);  //绘制最小外接矩形的中心点
            box1.points(rect);  //把最小外接矩形四个端点复制给rect数组
            for (int j = 0; j != 4; ++j) {
                cv::line(frame_plot, rect[j], rect[(j + 1) % 4], cv::Scalar(0, 255, 0), 2, 8);  //绘制最小外接矩形每条边
            }
            points.push_back(box1.center);//储存最小外接矩形中心点
            cv::Point2d c;  //圆心坐标
            double r = 0;   //半径
            LeastSquaresCircleFitting(points, c, r);//拟合圆
            cv::circle(frame_plot, c, r, cv::Scalar(0, 0, 255), 2, 8);//绘制圆
            cv::circle(frame_plot, c, 5, cv::Scalar(255, 0, 0), -1, 8);//绘制圆心

        }
    }
}

/**
 * @brief LeastSquaresCircleFitting
 * @param m_Points
 * @param Centroid
 * @param dRadius
 * @return
 */
int LeastSquaresCircleFitting(vector<cv::Point2d> &m_Points, cv::Point2d &Centroid, double &dRadius)//拟合圆函数
{
    if (!m_Points.empty())
    {
        int iNum = (int)m_Points.size();
        if (iNum < 3)	return 1;
        double X1 = 0.0;
        double Y1 = 0.0;
        double X2 = 0.0;
        double Y2 = 0.0;
        double X3 = 0.0;
        double Y3 = 0.0;
        double X1Y1 = 0.0;
        double X1Y2 = 0.0;
        double X2Y1 = 0.0;
        vector<cv::Point2d>::iterator iter;
        vector<cv::Point2d>::iterator end = m_Points.end();
        for (iter = m_Points.begin(); iter != end; ++iter)
        {
            X1 = X1 + (*iter).x;
            Y1 = Y1 + (*iter).y;
            X2 = X2 + (*iter).x * (*iter).x;
            Y2 = Y2 + (*iter).y * (*iter).y;
            X3 = X3 + (*iter).x * (*iter).x * (*iter).x;
            Y3 = Y3 + (*iter).y * (*iter).y * (*iter).y;
            X1Y1 = X1Y1 + (*iter).x * (*iter).y;
            X1Y2 = X1Y2 + (*iter).x * (*iter).y * (*iter).y;
            X2Y1 = X2Y1 + (*iter).x * (*iter).x * (*iter).y;
        }
        double C = 0.0;
        double D = 0.0;
        double E = 0.0;
        double G = 0.0;
        double H = 0.0;
        double a = 0.0;
        double b = 0.0;
        double c = 0.0;
        C = iNum * X2 - X1 * X1;
        D = iNum * X1Y1 - X1 * Y1;
        E = iNum * X3 + iNum * X1Y2 - (X2 + Y2) * X1;
        G = iNum * Y2 - Y1 * Y1;
        H = iNum * X2Y1 + iNum * Y3 - (X2 + Y2) * Y1;
        a = (H * D - E * G) / (C * G - D * D);
        b = (H * C - E * D) / (D * D - G * C);
        c = -(a * X1 + b * Y1 + X2 + Y2) / iNum;
        double A = 0.0;
        double B = 0.0;
        double R = 0.0;
        A = a / (-2);
        B = b / (-2);
        R = double(sqrt(a * a + b * b - 4 * c) / 2);
        Centroid.x = A;
        Centroid.y = B;
        dRadius = R;
        return 0;
    }
    else
        return 1;
    return 0;
}


