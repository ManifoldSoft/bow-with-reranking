//
//  main.cpp
//  vlfeat-bow-with-reranking
//
//  Created by willard on 8/24/15.
//  Copyright (c) 2015 wilard. All rights reserved.
// http://stackoverflow.com/questions/28606011/vlfeat-kdtree-setup-and-query
// http://mcreader-indep.googlecode.com/svn/!svn/bc/13/MCRindep/jni/

#include <iostream>

#include "file_io.h"
#include "feature_set.h"
#include "siftmatcher.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>

extern "C" {
#include "vl/generic.h"
#include "vl/sift.h"
#include "vl/kdtree.h"
#include "vl/random.h"
#include "vl/host.h"
}

using namespace std;

int main(int argc, const char * argv[]) {
    
    string descriptor_Query_path = "/Users/willard/codes/cpp/opencv-computer-vision/cpp/videosearch/indexer/local_features/test_images/img1.siftb";
    string descriptor_Object_path = "/Users/willard/codes/cpp/opencv-computer-vision/cpp/videosearch/indexer/local_features/test_images/img2.siftb";
    
    /*// 测试siftmatcher效果
     string name = "/Users/willard/Pictures/img1.jpg";
     siftmatcher* matcher;
     Image image;
     matcher->initializeImage(name, image);*/
    
    uint sift_desc_dim = 128;
    uint sift_keypoint_dim = 4;
    
    vector<FeatureSet*> test;
    
    FeatureSet* fQuerySet = readSIFTFile(descriptor_Query_path, sift_keypoint_dim, sift_desc_dim);
    FeatureSet* fObjectSet = readSIFTFile(descriptor_Object_path, sift_keypoint_dim, sift_desc_dim);
    
    // 创建kd树，做近似最近邻搜索，查询图像
    float* data = new float[128*(*fQuerySet).m_vDescriptors.size()];
    for (int i = 0; i < (*fQuerySet).m_vDescriptors.size(); i++)
        for (int j = 0; j < 128; j++)
            data[j+128*i] = (*fQuerySet).m_vDescriptors[i][j];
    VlKDForest* forest = vl_kdforest_new(VL_TYPE_FLOAT, 128, 1, VlDistanceL2);  //创建kd树对象，128维，1棵树
    forest->thresholdingMethod = VL_KDTREE_MEDIAN;
    vl_kdforest_build(forest, (*fQuerySet).m_vDescriptors.size(), data);
    
    // 目标图像
    float* fObjectData = new float[128*(*fObjectSet).m_vDescriptors.size()];
    for (int i = 0; i < (*fObjectSet).m_vDescriptors.size(); i++)
        for (int j = 0; j < 128; j++)
            fObjectData[j+128*i] = (*fObjectSet).m_vDescriptors[i][j];
    
    // 做最近查找 Searcher object
    VlKDForestSearcher* searcher = vl_kdforest_new_searcher(forest);
    vl_size neiborNum = 2;
    VlKDForestNeighbor neighbours[2];
    vl_size fObjectSetNum = (*fObjectSet).m_vDescriptors.size();
    vl_uint32* index1 = (vl_uint32 *)vl_malloc(sizeof(vl_uint32)*neiborNum*fObjectSetNum);
    float* dist = (float *)vl_malloc(sizeof(float)*neiborNum*fObjectSetNum);
    /* Query the first ten points for now */
    vector<pair<int, int>> matched_index;
    vector<Point2f> queryLoc, objectLoc;
    
    //vl_kdforest_query_with_array(searcher, index1, neiborNum, fObjectSetNum, dist, fObjectData);
    
    for(int i=0; i < (*fObjectSet).m_vDescriptors.size(); i++){
        vl_kdforestsearcher_query(searcher, neighbours, 2, fObjectData + 128*i);
        //auto nvisited = vl_kdforestsearcher_query(searcher, neighbours, 2, fObjectData + 128*i);
        cout << "最近邻：" << neighbours[0].index << " 距离：" << neighbours[0].distance << " " "次近邻：" << neighbours[1].index << " 距离：" << neighbours[1].distance << endl;
        if(neighbours[0].distance < 0.8*neighbours[1].distance){
            matched_index.push_back(pair<int, int>(neighbours[0].index, i));
            Point2f tmp1;
            tmp1.x = (*fQuerySet).m_vFrames[neighbours[0].index][0];
            tmp1.y = (*fQuerySet).m_vFrames[neighbours[0].index][1];
            queryLoc.push_back(tmp1);
            Point2f tmp2;
            tmp2.x = (*fObjectSet).m_vFrames[i][0];
            tmp2.y = (*fObjectSet).m_vFrames[i][1];
            objectLoc.push_back(tmp2);
        }
    }
    
    // 做最近查找 Searcher object
    /*VlKDForestSearcher* searcher = vl_kdforest_new_searcher(forest);
     VlKDForestNeighbor neighbours[2];
     vector<pair<int, int>> matched_index; //Query the first ten points for now
     vector<Point2f> queryLoc, objectLoc;
     for(int i=0; i < (*fObjectSet).m_vDescriptors.size(); i++){
     vl_kdforestsearcher_query(searcher, neighbours, 2, fObjectData + 128*i);
     //auto nvisited = vl_kdforestsearcher_query(searcher, neighbours, 2, fObjectData + 128*i);
     cout << "最近邻：" << neighbours[0].index << " 距离：" << neighbours[0].distance << " " "次近邻：" << neighbours[1].index << " 距离：" << neighbours[1].distance << endl;
     if(neighbours[0].distance < 0.8*neighbours[1].distance){
     matched_index.push_back(pair<int, int>(neighbours[0].index, i));
     Point2f tmp1;
     tmp1.x = (*fQuerySet).m_vFrames[neighbours[0].index][0];
     tmp1.y = (*fQuerySet).m_vFrames[neighbours[0].index][1];
     queryLoc.push_back(tmp1);
     Point2f tmp2;
     tmp2.x = (*fObjectSet).m_vFrames[i][0];
     tmp2.y = (*fObjectSet).m_vFrames[i][1];
     objectLoc.push_back(tmp2);
     }
     }*/
    
    //显示匹配的点对
    string imgfn = "/Users/willard/codes/cpp/opencv-computer-vision/cpp/videosearch/indexer/local_features/test_images/img1.jpg";
    string objFileName = "/Users/willard/codes/cpp/opencv-computer-vision/cpp/videosearch/indexer/local_features/test_images/img2.jpg";
    drawMatch(imgfn, objFileName, queryLoc, objectLoc);
    
    // 计算homography矩阵
    Mat mask;
    vector<Point2f> queryInliers;
    vector<Point2f> sceneInliers;
    //Mat H = findFundamentalMat(queryLoc, objectLoc, mask, CV_FM_RANSAC);
    Mat H = findHomography(queryLoc, objectLoc, CV_RANSAC, 10, mask);
    int inliers_cnt = 0, outliers_cnt = 0;
    for (int j = 0; j < mask.rows; j++){
        if (mask.at<uchar>(j) == 1){
            queryInliers.push_back(queryLoc[j]);
            sceneInliers.push_back(objectLoc[j]);
            inliers_cnt++;
        }else {
            outliers_cnt++;
        }
    }
    //显示剔除误配点对后的匹配点对
    drawMatch(imgfn, objFileName, queryInliers, sceneInliers);
    
    
    fQuerySet->print();
    
    // Clean up
    if (fQuerySet) {
        delete fQuerySet;
    }
    
    return EXIT_SUCCESS;
}