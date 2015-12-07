//
//  test_extract.cpp
//  vlfeat-bow-with-reranking
//
//  Created by willard on 11/7/15.
//  Copyright © 2015 wilard. All rights reserved.
//

#include "test_extract.h"
#include "sift_extractor.h"
#include "utils.h"

using namespace cv;
using namespace arma;

void test_extract(){
    string image_path = "/Users/willard/codes/cpp/opencv-computer-vision/vlfeat-bow-with-reranking/vlfeat-bow-with-reranking/test_image.jpg";
    vector<float*> keypoints;
    vector<float*> descriptors;
    unsigned int number_desc;
    int verbose = 1;
    bool display_image = true;
    bool divide_512 = false;
    
    bool err = sift_keypoints_and_descriptors(image_path, divide_512, verbose, display_image, keypoints, descriptors, number_desc);
    
    if (err) {
        cout << number_desc << " descriptors were detected successfully" << endl;
        
    }
    else {
        cout << "Descriptors were NOT detected successfully" << endl;
    }
};


