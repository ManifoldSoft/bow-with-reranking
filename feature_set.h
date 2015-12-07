//
//  feature_set.h
//  vlfeat-bow-with-reranking
//
//  Created by willard on 8/24/15.
//  Copyright (c) 2015 wilard. All rights reserved.
//

#ifndef __vlfeat_bow_with_reranking__feature_set__
#define __vlfeat_bow_with_reranking__feature_set__

#include <stdio.h>

#ifndef _FEATURE_SET_H_
#define _FEATURE_SET_H_

#include <vector>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short ushort;

/* ----------------------------------------------------------------
 * class: FeatureSet
 * ----------------------------------------------------------------
 * Contains a set of features.
 * ----------------------------------------------------------------
 */
class FeatureSet
{
public:
    
    /* ----------------------------------------------------------------
     * Constructor: FeatureSet
     * ----------------------------------------------------------------
     * Creates a FeatureSet object.
     * ----------------------------------------------------------------
     */
    FeatureSet(unsigned int nDescriptorLength, unsigned int nFrameLength);  // 构造函数1
    FeatureSet(FeatureSet* pOther);  // 构造函数2
    
    /* ----------------------------------------------------------------
     * Destructor: ~FeatureSet
     * ----------------------------------------------------------------
     * Destroys a FeatureSet object.
     * ----------------------------------------------------------------
     */
    ~FeatureSet();
    
    /* ----------------------------------------------------------------
     * function: addFeature
     * ----------------------------------------------------------------
     * Adds a new feature to the set.
     * ----------------------------------------------------------------
     */
    void addFeature(float* pDescriptor, float* pFrame);
    
    /* ----------------------------------------------------------------
     * Method: print
     * ----------------------------------------------------------------
     * Print a summary of the feature set.
     * ----------------------------------------------------------------
     */
    void print();
    
    // Member variables
    vector<float*> m_vDescriptors;
    vector<float*> m_vFrames;
    unsigned int m_nDescriptorLength;  // SIFT特征点的维度
    unsigned int m_nFrameLength;  // SIFT关键点的维度
    unsigned int m_nNumFeatures;  // SIFT特征点数目
    
};

#endif


#endif /* defined(__vlfeat_bow_with_reranking__feature_set__) */
