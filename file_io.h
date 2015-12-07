//
//  file_io.h
//  vlfeat-bow-with-reranking
//
//  Created by willard on 8/24/15.
//  Copyright (c) 2015 wilard. All rights reserved.
//

#ifndef __vlfeat_bow_with_reranking__file_io__
#define __vlfeat_bow_with_reranking__file_io__

#include <stdio.h>

#ifndef FILE_IO_H
#define FILE_IO_H

#include <algorithm>
#include <string>

#include "feature_set.h"

using namespace std;

const int MAX_NUM_FEATURES = 5000;
//const float DESC_DIVISOR = 1.0;
const float DESC_DIVISOR = 512.0;

FeatureSet* readSIFTFile(string sFilename, uint nFrameLength, uint nDescLength);

#endif

#endif /* defined(__vlfeat_bow_with_reranking__file_io__) */
