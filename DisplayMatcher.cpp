//
//  DisplayMatcher.cpp
//  vlfeat-bow-with-reranking
//
//  Created by willard on 8/24/15.
//  Copyright (c) 2015 wilard. All rights reserved.
//
// http://mcreader-indep.googlecode.com/svn/!svn/bc/13/MCRindep/jni/MCRindep.cpp
// http://stackoverflow.com/questions/28606011/vlfeat-kdtree-setup-and-query


#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>

#include<string>
#include<iostream>
#include<sstream>
#include<fstream>
#include<iomanip>
#include<cmath>
#include<algorithm>
#include<vector>
using namespace std;

extern "C" {
#include "vl/generic.h"
#include "vl/sift.h"
#include "vl/kdtree.h"
}
#include "Image.h"
#include "Exception.h"

#include "DisplayMatcher.h"


////////////////Constructor/////////////////////
/*
 * 构造函数
 */
DisplayMatcher::DisplayMatcher()
{
    testDescriptors = new float[128*MAXDESCRIPTORS];  //描述子以一维数组保存
    libraryDescriptors = new float[128*MAXDESCRIPTORS];
    numTestDescriptors = 0;
    numLibDescriptors = 0;
    numMatches = 0;
}

/////////////////////////////////////////////////////

////////////////Deconstructor/////////////////////
DisplayMatcher::~DisplayMatcher()
{
    delete testDescriptors;
    delete libraryDescriptors;
    vl_kdforest_delete(forest);
}

/////////////////////////////////////////////////////


///////////////////processTestImage//////////////////
void DisplayMatcher::processTestImage( string imageName )
{
    vl_sift_pix* testGray = initializeImage( imageName, testImage);			//Initialize the first image
    numTestDescriptors = sift(testDescriptors, testGray, testImage, true);  //sift and store the descriptors in testDescr, and return the # of descriptors
    normalize(testDescriptors, numTestDescriptors);				//normalize the descriptor values
}

/////////////////////////////////////////////////////

///////////////////processLibraryImage//////////////////
/* 函数名称：processLibraryImage()提取sift特征点
 * 重要参数：
 * libName：图像文件名
 * numLibDescriptors：特征点数
 * libraryDescriptors：特征存储在该变量中
 *
 */
void DisplayMatcher::processLibraryImage(string libName)
{
    vl_sift_pix* libGray = initializeImage(libName, libImage);			//Initialize the Library image
    numLibDescriptors = sift(libraryDescriptors, libGray, libImage, false);    //sift and store the descriptors in LibraryDescr, and return the # of descriptors
    normalize(libraryDescriptors, numLibDescriptors);				//normalize the descriptor values
}

/////////////////////////////////////////////////////

//////////////////////buildForest////////////////////
/* 函数名称：buildForest()，建立kd树
 * 关于vl_kdforest_new()距离的选取可以参考链接 http://www.vlfeat.org/api/mathop_8h.html#af7397bb42d71000754eafba9458b09acab44b9322465d703274acfa21690de9fd
 */
void DisplayMatcher::buildForest()
{
    VlKDForest* forest = vl_kdforest_new(VL_TYPE_FLOAT, 128, 1, VlDistanceL2);  //创建kd树对象，128维，1棵树
    forest->thresholdingMethod = VL_KDTREE_MEDIAN;
    vl_kdforest_build(forest, numLibDescriptors, libraryDescriptors);
}

/////////////////////////////////////////////////////

/////////////////////findMatches/////////////////////
int DisplayMatcher::findMatches()
{
    int matches = 0;
    //vl_kdforest_query runs a query on a SINGLE descriptor and returns the nearest neighbors
    _VlKDForestNeighbor* neighbors = new _VlKDForestNeighbor[numTestDescriptors];
    //_VlKDForestNeighbor* neighborMatches[1000];
    for( int i=0; i<numTestDescriptors; i++ )
    {
        vl_kdforest_query(forest, neighbors, 1, testDescriptors+i*128);
        if( (neighbors)->distance < THRESHOLD)
        {
            this->indexPairs[matches*2] = i;
            this->indexPairs[matches*2+1] = neighbors->index;
            matches++;
        }
    }
    delete [] neighbors;
    //delete [] neighborMatches;
    this->numMatches = matches;
    return matches;
}

vl_sift_pix* DisplayMatcher::initializeImage( string name, Image& image )
{
    ifstream ifs(name.c_str(), ios_base::in | ios_base::binary) ;
    if(!ifs) {
        throw Exception("Could not open a file") ;
    }
    ifs>>image;
    vl_sift_pix* gray = new vl_sift_pix[image.getHeight()*image.getWidth()];
    convertToGrayscale(image, gray );
    return gray;
}

void DisplayMatcher::normalize( float* descr, int numDescr )
{
    for ( int i=0; i < numDescr*128; i++ )			//Normalize the values
    {
        float x = 512.0 * descr[i];
        x = (x < 255.0) ? x : 255.0 ;
        descr[i] = x;
    }
}

void DisplayMatcher::convertToGrayscale(Image img, vl_sift_pix *gray )
{
    if( img.getDataSize() == img.getHeight()*img.getWidth() )		//already grayscale, since pixelsize = 1
        for ( int i=0; i<img.getHeight()*img.getWidth(); i++ )
            gray[i] = (float)*(img.getDataPt()+i);
    else															//converting rgb to grayscale
        for ( int i=0; i < img.getHeight()*img.getWidth(); i++ )
            gray[i] = (float)( 0.3 * *(img.getDataPt()+3*i) + 0.59 * *(img.getDataPt()+3*i+1) + 0.11 * *(img.getDataPt()+3*i+2) );
}

/* 函数名称：sift()
 *
 * 函数返回值：返回sift的特征点数
 */
int DisplayMatcher::sift(float* d, float* pixels, Image I, bool testImage)  //returns the number of keypoints
{
    VlSiftFilt *sift = vl_sift_new(I.getWidth(), I.getHeight(), -1, 3, 0);
    vl_sift_process_first_octave(sift, pixels);  //process over the first octave
    int count = 0;
    do
    {
        vl_sift_detect(sift);
        const VlSiftKeypoint* key = vl_sift_get_keypoints(sift);
        for(int i=0; i<sift->nkeys; i++)
        {
            double angles[4];
            int numOrients = vl_sift_calc_keypoint_orientations(sift,angles,key+i);
            for(int j=0;j<numOrients; j++)
            {
                vl_sift_calc_keypoint_descriptor(sift,(d+128*count),key+i,angles[j]);		//calculate the descriptor
                if( testImage )
                {
                    xyPairsTest[2*count] = (key+i)->x;
                    xyPairsTest[2*count+1] = (key+i)->y;
                }
                else
                {
                    xyPairsLibrary[2*count] = (key+i)->x;
                    xyPairsLibrary[2*count+1] = (key+i)->y;
                }
                count++;
            }
        }
    }while( vl_sift_process_next_octave(sift) != VL_ERR_EOF );		//iterate over the rest of the octaves
    vl_sift_delete(sift);
    return count;
}

void DisplayMatcher::insertMarkers( Image& I, int* xyPairs, int numPairs)
{
    for ( int i=0; i<numPairs; i++ )
    {
        int image1x = xyPairs[2*i];
        int image1y = xyPairs[2*i+1];
        unsigned char* pt = I.getPixelPt( image1x, image1y);
        pt[0] = 255;
        pt[1] = 0;
        pt[2] = 0;
    }
}

void DisplayMatcher::outputMatches(string testImageName, string libImageName)
{
    int* xyPairsTestMatches = new int[2*numMatches];
    int* xyPairsLibMatches = new int[2*numMatches];
    
    for(int i=0; i<numMatches; i++)
    {
        xyPairsTestMatches[2*i] = xyPairsTest[2 * this->indexPairs[2*i] ];
        xyPairsTestMatches[2*i+1] = xyPairsTest[2 * this->indexPairs[2*i] + 1];
        
        xyPairsLibMatches[2*i] = xyPairsLibrary[2 * this->indexPairs[2*i+1] ];
        xyPairsLibMatches[2*i+1] = xyPairsLibrary[2 * this->indexPairs[2*i+1] + 1];
    }
    
    insertMarkers(testImage, xyPairsTestMatches, numMatches);
    insertMarkers(libImage, xyPairsLibMatches, numMatches);
    
    outputImage(testImage, testImageName);
    cout<< "Wrote out the test image with matches to " << testImageName << endl;
    outputImage(libImage, libImageName);
    cout<< "Wrote out the matching library image with matches to " << libImageName <<endl;
}

void DisplayMatcher::outputImage(Image& testImage, string testImageName)
{
    ofstream ofile(testImageName.c_str(), ios_base::out | ios_base::binary);
    ofile<<testImage;
    ofile.close();
}
