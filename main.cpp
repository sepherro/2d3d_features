#include <iostream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int circle_coeffs[16][2] = {    {0, 3}, {1, 3}, {2, 2}, {3, 1},
                                {3, 0}, {3, -1}, {2, -2}, {1, -3},
                                {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
                                {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3} };

vector<Point3f> normalized_bresenham (16);

Point3f normalize_vector(Point3f point)
{
    Point3f result;

    float norm = sqrtf(point.x * point.x + point.y * point.y + point.z * point.z );
    result.x = point.x/norm;
    result.y = point.y/norm;
    result.z = point.z/norm;

    return result;
}

Point3f normalize_point(int x, int y)
{
    Point3f temp_pt( (x - 319.5)/525.0, (y - 239.5)/525.0, 1 );
    return temp_pt;
}

int main()
{
    FastFeatureDetector fast_detector(40);              // threshold as parameter -- alternatively, to retain best N KeyPointsFilter::retainBest(keypoints, N);
    vector<KeyPoint> keypoints, keypoints_scaled;
    KeyPoint kpt, kpt_n;

    Mat rgb_image=imread("C:\\Users\\Marek\\qtprojects\\images\\rgb.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat depth_image=imread("C:\\Users\\Marek\\qtprojects\\images\\depth.png", CV_LOAD_IMAGE_ANYDEPTH);
    Mat depth_image_mask;
    convertScaleAbs(depth_image, depth_image_mask, 5000, 0);

    fast_detector.detect(rgb_image, keypoints, depth_image_mask);

    drawKeypoints(rgb_image, keypoints, rgb_image);

    imshow("window", rgb_image);

    imshow("dwindow", depth_image);

    depth_image.convertTo(depth_image, CV_32F);

    for(unsigned int i = 0; i < keypoints.size(); i++)
    {

        kpt = keypoints[i];

        float lambdas[16];

        float depth_center = depth_image.at<float>(kpt.pt.y, kpt.pt.x);

       // divide the depths of the points on the Bresenham circle by the depth of the central point -> normalized depth
        for(unsigned int i = 0; i < 16; i++)
        {
            lambdas[i] = depth_image.at<float>(kpt.pt.y + circle_coeffs[i][0], kpt.pt.x + circle_coeffs[i][1]) / depth_center;
        }

        // normalize the coordinates of points

        Point3f pt_c = normalize_point( kpt.pt.y, kpt.pt.x );


        for(unsigned int i = 0; i < 16; i++)
        {
            Point3f temp_pt;
            temp_pt = normalize_point( kpt.pt.y + circle_coeffs[i][0], kpt.pt.x + circle_coeffs[i][1] ) * lambdas[i];
            normalized_bresenham.push_back( temp_pt );
        }

        for(unsigned int i = 0; i < 16; i++)
        {   //diff
            normalized_bresenham[i] = normalized_bresenham[i] - pt_c;
        }

//        pt_00 = pt_00 - pt_c;
//        pt_01 = pt_01 - pt_c;
//        pt_02 = pt_02 - pt_c;
//        pt_03 = pt_03 - pt_c;
//        pt_04 = pt_04 - pt_c;
//        pt_05 = pt_05 - pt_c;
//        pt_06 = pt_06 - pt_c;
//        pt_07 = pt_07 - pt_c;
//        pt_08 = pt_08 - pt_c;
//        pt_09 = pt_09 - pt_c;
//        pt_10 = pt_10 - pt_c;
//        pt_11 = pt_11 - pt_c;
//        pt_12 = pt_12 - pt_c;
//        pt_13 = pt_13 - pt_c;
//        pt_14 = pt_14 - pt_c;
//        pt_15 = pt_15 - pt_c;

        waitKey();


    }




    waitKey();
    return 0;
}



