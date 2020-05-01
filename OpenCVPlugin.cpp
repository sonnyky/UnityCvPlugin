//
//  OpenCVPlugin.cpp
//  UnityCvPlugin
//
//  Created by sonny on 2019/10/07.
//  Copyright © 2019年 sonny. All rights reserved.
//

#include <stdio.h>
#include <string>
#include <iostream>
#include <map>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;

extern "C" {
    void _SaveBlackAndWhite(unsigned char * bytes, int rows, int cols);
    void _DetectOuterHull(unsigned char * bytes, int rows, int cols);
    float _CompareStructureSimilarity(unsigned char * bytesRef, unsigned char * toCompare, int rows, int cols);
    void _CompareSimilarityWithFeatures(unsigned char * bytesRef, unsigned char * toCompare, int rows, int cols);
    void _TransformImage(unsigned char * original, unsigned char * target
    , int rows, int cols,
    float rotateAngle = 0, float scale = 1, int transX = 0, int transY = 0);
}

void _SaveBlackAndWhite(unsigned char * bytes, int rows, int cols){
    
    Mat img(rows, cols, CV_8UC4);
    memcpy(img.data, bytes, rows * cols * 4);
    
    Mat converted(rows, cols, CV_8UC4);
    cvtColor(img, converted, COLOR_RGBA2BGRA);
    
    flip(converted, converted, 0);
    
    imwrite("bw.png", converted);
    
}

Rect detectImageRegion(Mat img, int rows, int cols, int type){
    int dilation_elem = 0;
    int dilation_size = 5;

    Mat converted(rows, cols, CV_8UC4);
    cvtColor(img, converted, COLOR_RGBA2BGRA);

    Mat edges;
    Canny(converted, edges, 100, 200);

    int dilation_type = 0;
    if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
    else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
    else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement( dilation_type,
                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                        Point( dilation_size, dilation_size ) );
    dilate( edges, edges, element );
    //Detect contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours( edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    int largest_area=0;
    int largest_contour_index=0;

    Mat drawing = Mat::zeros( edges.size(), CV_8UC3 );

    for(int i = 0; i < contours.size(); i++)
    {
       double a=contourArea(contours[i],false);
       if(a>largest_area)
       {
           largest_area=a;
           largest_contour_index=i;
       }
    }
    vector<Point> hullPoints;
    convexHull(contours[largest_contour_index],hullPoints);
    Rect boundRect;
    boundRect = boundingRect(contours[largest_contour_index]);
    return boundRect;
}

Mat rotateImage(Mat input, double angle){
    int height = input.rows;
    int width = input.cols;
    int centerX = width/2;
    int centerY = height/2;
    
    Mat rotation = getRotationMatrix2D(Point2f(centerX, centerY), angle, 1.0);
    double cos = abs(rotation.at<double>(0,0));
    double sin = abs(rotation.at<double>(0,1));
    
    // new image size
    int newW = int((height * sin) + (width * cos));
    int newH = int((height * cos) + (width * sin));
    
    rotation.at<double>(0,2) += (newW / 2) - centerX;
    rotation.at<double>(1,2) += (newH / 2) - centerY;
    Mat rotated(newW, newH, input.type());
    warpAffine(input, rotated, rotation, Size(newW, newH));
    return rotated;
}

void _TransformImage(unsigned char * original, unsigned char * target
                     , int rows, int cols,
                     float rotateAngle, float scale, int transX, int transY){
    
    // Prepare local variables to perform processing on
    Mat img(rows, cols, CV_8UC4);
    memcpy(img.data, original, rows * cols * 4);
        
    // Prepare another image of the same size as input and set all pixels to white
    // This will be the final output image
    Mat transformed(rows, cols, CV_8UC4);
    transformed = Scalar(255, 255, 255, 255);
    
    // Prepare another image as temporary image
    Mat temp;
    int newRows = rows * scale;
    int newCols = cols * scale;
    if(scale != 1){
        resize(img, temp, Size(newCols, newRows));
        // after resizing image size is changed so detect image region
        Rect regionAfterResize = detectImageRegion(temp, temp.rows, temp.cols, CV_8UC4);
        
        // Create a new image from the ROI
        temp = temp(regionAfterResize);
    }
    
    if(rotateAngle != 0){
        temp = rotateImage(temp, rotateAngle);
    }
    
    temp.copyTo(transformed(cv::Rect(transX,transY,temp.cols, temp.rows)));
    memcpy(target, transformed.data, rows * cols * 4);
}

vector<Point> contoursConvexHull( vector<vector<Point> > contours )
{
    vector<Point> result;
    vector<Point> pts;
    for ( size_t i = 0; i< contours.size(); i++)
        for ( size_t j = 0; j< contours[i].size(); j++)
            pts.push_back(contours[i][j]);
    convexHull( pts, result );
    return result;
}

void _DetectOuterHull(unsigned char * bytes, int rows, int cols){
    int dilation_elem = 0;
    int dilation_size = 5;
    
    Mat img(rows, cols, CV_8UC4);
    memcpy(img.data, bytes, rows * cols * 4);
    
    Mat converted(rows, cols, CV_8UC4);
    cvtColor(img, converted, COLOR_RGBA2BGRA);
    
    Mat edges;
    Canny(converted, edges, 100, 200);
    
    int dilation_type = 0;
    if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
    else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
    else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement( dilation_type,
                         Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                         Point( dilation_size, dilation_size ) );
    dilate( edges, edges, element );
    imwrite("edges.jpg", edges);

    //Detect contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours( edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

    Scalar colour = Scalar(255, 0, 255);
    Scalar hullColor = Scalar(255, 255, 0);
    Scalar rectColor = Scalar(0, 0, 255);
    
    int largest_area=0;
    int largest_contour_index=0;

    Mat drawing = Mat::zeros( edges.size(), CV_8UC3 );

    for(int i = 0; i < contours.size(); i++)
    {
        double a=contourArea(contours[i],false);
        if(a>largest_area)
        {
            largest_area=a;
            largest_contour_index=i;
        }
        drawContours(drawing, contours, i, colour, 1, 8, hierarchy, 0, Point());
    }
    
    vector<Point> hullPoints;
    convexHull(contours[largest_contour_index],hullPoints);

     drawContours(drawing, vector<vector<Point>> {hullPoints}, 0, hullColor, 3, 8);

    Rect boundRect;
    boundRect = boundingRect(contours[largest_contour_index]);
    rectangle( drawing, boundRect.tl(), boundRect.br(), rectColor, 2 );
    imwrite("contours.jpg", drawing);
}

float _CompareStructureSimilarity(unsigned char * bytesRef, unsigned char * toCompare, int rows, int cols){
    
    
    // Create opencv images from the bytes
    Mat refImg(rows, cols, CV_8UC4);
    memcpy(refImg.data, bytesRef, rows * cols * 4);
    
    Mat imgToCompare(rows, cols, CV_8UC4);
    memcpy(imgToCompare.data, toCompare, rows * cols * 4);

    
    // Create grayscale versions
    Mat grayRefImg(rows, cols, CV_8UC1);
    Mat grayToCompare(rows, cols, CV_8UC1);
    cvtColor(refImg, grayRefImg, COLOR_BGRA2GRAY);
    cvtColor(imgToCompare, grayToCompare, COLOR_BGRA2GRAY);
    
    adaptiveThreshold(grayRefImg, grayRefImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 2);

    adaptiveThreshold(grayToCompare, grayToCompare, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 2);
    
    // Calculate image similarity from the thresholded grayscale images, internally using Hu Moments
    double res = matchShapes(grayRefImg,grayToCompare,CONTOURS_MATCH_I2,0);
    auto resF = static_cast<float>(res);
    
    return resF;
}


void _CompareSimilarityWithFeatures(unsigned char * bytesRef, unsigned char * toCompare, int rows, int cols){
    // Create opencv images from the bytes
    Mat refImg(rows, cols, CV_8UC4);
    memcpy(refImg.data, bytesRef, rows * cols * 4);
    
    Mat imgToCompare(rows, cols, CV_8UC4);
    memcpy(imgToCompare.data, toCompare, rows * cols * 4);
    
    std::vector<cv::KeyPoint> detectedKeypoints;
    cv::Mat objectDescriptors;
    
    // Extract data
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    f2d->detect( refImg, keypoints_1 );
    f2d->detect( imgToCompare, keypoints_2 );
    
    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute( refImg, keypoints_1, descriptors_1 );
    f2d->compute( imgToCompare, keypoints_2, descriptors_2 );
    
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_1, descriptors_2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.3f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    
    //-- Draw matches
    Mat img_matches;
    drawMatches( refImg, keypoints_1, imgToCompare, keypoints_2, good_matches, img_matches, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
}
