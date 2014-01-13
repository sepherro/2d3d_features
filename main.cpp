#include <iostream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "2d3d_features.hpp"

using namespace cv;







int main()
{
    // use your own images :)
    Mat rgb_image=imread("C:\\Users\\Marek\\Desktop\\obrazy_kinect\\rgb_000.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat depth_image=imread("C:\\Users\\Marek\\Desktop\\obrazy_kinect\\depth_000.png", CV_LOAD_IMAGE_ANYDEPTH);

    Mat rgb_copy = rgb_image.clone();

    vector<KeyPoint> keypoints_okay;

    //parameters:
    //1. rgb image, 2. corresponding depth image, 3. FAST detector threshold, 4. flatness threshold (1 - completely flat, 0 - no constraints on flatness)
    keypoints_okay = detect_rgbd_features(rgb_image, depth_image, 40, 0.8);

    drawKeypoints(rgb_image,keypoints_okay, rgb_image);
    imshow("results_filtered", rgb_image);

    keypoints_okay = detect_rgbd_features(rgb_image, depth_image, 40, 0.0);

    drawKeypoints(rgb_copy,keypoints_okay, rgb_copy);
    imshow("results_unfiltered", rgb_copy);


    waitKey();
    return 0;
}



