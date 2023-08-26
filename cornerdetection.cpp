#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main() {
    cv::Mat img = cv::imread("download.jpeg");

    if (img.empty()) {
        std::cout << "Error loading image!" << std::endl;
        return -1;
    }

    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray_img, corners, 100, 0.01, 10);

    for (size_t i = 0; i < corners.size(); ++i) {
        int x = static_cast<int>(corners[i].x);
        int y = static_cast<int>(corners[i].y);
        cv::circle(img, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), 2);
    }

    cv::imwrite("img1.png", img);

    return 0;
}
