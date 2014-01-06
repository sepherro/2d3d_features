#include <iostream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

#define FLATNESS_THRESHOLD 0.6

int circle_coeffs[16][2] = {    {0, 3}, {1, 3}, {2, 2}, {3, 1},
                                {3, 0}, {3, -1}, {2, -2}, {1, -3},
                                {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
                                {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3} };

vector<Point3f> normalized_bresenham (16);

Point3f temp_pt;

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
    vector<KeyPoint> keypoints;
    KeyPoint kpt;
    unsigned int kickout;

    Mat rgb_image=imread("C:\\Users\\Marek\\qtprojects\\images\\rgb.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat depth_image=imread("C:\\Users\\Marek\\qtprojects\\images\\depth.png", CV_LOAD_IMAGE_ANYDEPTH);
    Mat depth_image_mask;
    convertScaleAbs(depth_image, depth_image_mask, 5000, 0);

    Mat match_img_all = rgb_image.clone();
    Mat match_img_okay = rgb_image.clone();
    Mat match_img_rejects = rgb_image.clone();



    fast_detector.detect(rgb_image, keypoints, depth_image_mask);

    drawKeypoints(match_img_all, keypoints, match_img_all);

    imshow("window", match_img_all);
    imshow("dwindow", depth_image);



    depth_image.convertTo(depth_image, CV_32F);

    vector<KeyPoint> keypoints_rejects;
    vector<KeyPoint> keypoints_okay;


    for(unsigned int j = 0; j < keypoints.size(); j++)
    {

        kpt = keypoints[j];

        float lambdas[16];

        float depth_center = depth_image.at<float>(kpt.pt.y, kpt.pt.x);

        // divide the depths of the points on the Bresenham circle by the depth of the central point -> normalized depth
        for(unsigned int i = 0; i < 16; i++)
        {
            lambdas[i] = depth_image.at<float>(kpt.pt.y + circle_coeffs[i][0], kpt.pt.x + circle_coeffs[i][1]) / depth_center;
        }

        // normalize the coordinates of the central point
        Point3f pt_c = normalize_point( kpt.pt.y, kpt.pt.x );

        // multiply by the computed scaling coefficients
        for(unsigned int i = 0; i < 16; i++)
        {
            Point3f temp_pt;
            temp_pt = normalize_point( kpt.pt.y + circle_coeffs[i][0], kpt.pt.x + circle_coeffs[i][1] ) * lambdas[i];
            //cout << temp_pt << endl;
            normalized_bresenham[i] = temp_pt;
        }
        //cout << lambdas;
        //cout << normalized_bresenham << endl << endl;

        // subtract the central point vector -> results are vectors connecting the central point to bresenham points
        for(unsigned int i = 0; i < 16; i++)
        {   //diff
            temp_pt = normalized_bresenham[i] - pt_c;
            normalized_bresenham[i] = temp_pt;
        }
        //cout << normalized_bresenham << endl << endl;

        // normalize the vectors to unit length
        for(unsigned int i = 0; i < 16; i++)
        {
            temp_pt = normalize_vector( normalized_bresenham[i] );
            normalized_bresenham[i] = temp_pt;
        }
        //cout << normalized_bresenham << endl << endl;


        kickout = 0;
        // decide whether or not keep the keypoint based on the flatness established by the dot product of opposing points
        for(unsigned int i = 0; i < 8; i++)
        {

            //cout << normalized_bresenham[i] << endl;
            //cout << normalized_bresenham[i+1] << endl;
            //cout << abs( normalized_bresenham[i].dot( normalized_bresenham[i+8]) ) << endl;
            if( abs( normalized_bresenham[i].dot( normalized_bresenham[i+8] ) ) < FLATNESS_THRESHOLD )
            {
                kickout = 1; // figure out how to delete keypoints not meeting criteria from the list
            }

        }

        if( kickout )
        {
            keypoints_rejects.push_back(kpt);
        }
        else
        {
            keypoints_okay.push_back(kpt);
        }


        //waitKey();
    }

    drawKeypoints(match_img_okay, keypoints_okay, match_img_okay);
    imshow("okay", match_img_okay);

    drawKeypoints(match_img_rejects, keypoints_rejects, match_img_rejects);
    imshow("rejected", match_img_rejects);

    cout << "okay: " << keypoints_okay.size() <<", rejected: " << keypoints_rejects.size() << endl;


    waitKey();
    return 0;
}



