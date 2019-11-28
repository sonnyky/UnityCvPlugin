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
    int _TestFunction_Internal();
    void _SaveBlackAndWhite(unsigned char * bytes, int rows, int cols, int type);
    float _TestGetImageShapeSimilarity();
    float _CompareStructureSimilarity(unsigned char * bytesRef, unsigned char * toCompare, int rows, int cols, int type);
    float _CompareSimilarityWithFeatures(unsigned char * bytesRef, unsigned char * toCompare, int rows, int cols, int type);
}

int _TestFunction_Internal() {
    return 12345;
}

float _TestGetImageShapeSimilarity(){
    
    float res = 0.52f;
    
    return res;
}

void _SaveBlackAndWhite(unsigned char * bytes, int rows, int cols, int type){
    
    Mat img(rows, cols, CV_8UC4);
    memcpy(img.data, bytes, rows * cols * 4);
    
    Mat converted(rows, cols, CV_8UC4);
    cvtColor(img, converted, COLOR_RGBA2BGRA);
    
    flip(converted, converted, 0);
    
    imwrite("bw.jpg", converted);
    
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

float AddCompareSimilarityWithFeatures(Mat refImg, Mat imgToCompare){
    
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
    int goodMatches = 0;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            goodMatches++;
        }
    }
    
    return (float) goodMatches;
}


float _CompareStructureSimilarity(unsigned char * bytesRef, unsigned char * toCompare, int rows, int cols, int type){
    
    
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
    
    // Specify type to add more reliability with feature detection. Runtime will be severely impacted.
    if(resF > 0.0001 && resF < 0.001 && type == 2){
        resF = AddCompareSimilarityWithFeatures(refImg, imgToCompare);
    }
    
    return resF;
}


float _CompareSimilarityWithFeatures(unsigned char * bytesRef, unsigned char * toCompare, int rows, int cols, int type){
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
    
    // Just return something as dummy
    return 0;
}
