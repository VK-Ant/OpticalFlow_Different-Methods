#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

void calculateOpticalFlow(cv::VideoCapture& cap) {
    cv::Mat prevFrame, currentFrame;
    cap >> prevFrame;
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);

    cv::Mat flow, flowColor;

    while (cap.read(currentFrame)) {
        cv::cvtColor(currentFrame, currentFrame, cv::COLOR_BGR2GRAY);

        cv::calcOpticalFlowFarneback(prevFrame, currentFrame, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        cv::cvtColor(prevFrame, flowColor, cv::COLOR_GRAY2BGR);
        for (int y = 0; y < flow.rows; y += 10) {
            for (int x = 0; x < flow.cols; x += 10) {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                cv::line(flowColor, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), cv::Scalar(0, 255, 0));
                cv::circle(flowColor, cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), 1, cv::Scalar(0, 255, 0), -1);
            }
        }

        cv::imshow("Lucas-Kanade Optical Flow", flowColor);
        
        if (cv::waitKey(30) == 27) {
            break;
        }

        prevFrame = currentFrame.clone();
    }
}

int main() {
    cv::VideoCapture cap("forest.mp4");
    
    if (!cap.isOpened()) {
        std::cout << "Error opening video file!" << std::endl;
        return -1;
    }

    calculateOpticalFlow(cap);

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
