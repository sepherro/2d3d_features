#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

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



vector<KeyPoint> detect_rgbd_features(Mat rgb_image, Mat depth_image, int fast_threshold, float flatness_threshold)
{

    //shift coefficients for FAST Bresenham circle point coordinates
    int circle_coeffs[16][2] = {    {0, 3}, {1, 3}, {2, 2}, {3, 1},
                                    {3, 0}, {3, -1}, {2, -2}, {1, -3},
                                    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
                                    {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3} };
    //raw keypoints detected by masked FAST
    vector<KeyPoint> raw_keypoints;
    //function result - filtered keypoints
    vector<KeyPoint> filtered_keypoints;
    //create temportary Point3f storage
    Point3f temp_pt;
    //create temporary keypoint storage
    KeyPoint kpt;
    //create fast detector with defined threshold
    FastFeatureDetector fast_detector(fast_threshold);
    //create storage for the detector mask
    Mat depth_image_mask;
    //normalized depths of points on the Bresenham circle
    vector<Point3f> normalized_bresenham (16);

    //create detector mask
    convertScaleAbs(depth_image, depth_image_mask, 5000, 0);
    //create fast features using mask (features not detected if depth=0)
    fast_detector.detect(rgb_image, raw_keypoints, depth_image_mask);
    //convert for 'at' compatibility
    depth_image.convertTo(depth_image, CV_32F);

    //for each detected keypoint do
    for(unsigned int j = 0; j < raw_keypoints.size(); j++)
    {
        kpt = raw_keypoints[j];

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
            temp_pt = normalize_point( kpt.pt.y + circle_coeffs[i][0], kpt.pt.x + circle_coeffs[i][1] ) * lambdas[i];
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


        unsigned int kickout = 0;
        // decide whether or not keep the keypoint based on the flatness established by the dot product of opposing points
        for(unsigned int i = 0; i < 8; i++)
        {

            if( abs( normalized_bresenham[i].dot( normalized_bresenham[i+8] ) ) < flatness_threshold )
            {
                kickout = 1; // figure out how to delete keypoints not meeting criteria from the list
            }

        }

        if( !kickout )
        {
            filtered_keypoints.push_back(kpt);
        }


    }
    return filtered_keypoints;

}
