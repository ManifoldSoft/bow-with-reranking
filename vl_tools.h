//
//  vl_tools.h
//  vlfeat-bow-with-reranking
//
//  Created by willard on 9/4/15.
//  Copyright (c) 2015 wilard. All rights reserved.
//

#ifndef __vlfeat_bow_with_reranking__vl_tools__
#define __vlfeat_bow_with_reranking__vl_tools__

#include <stdio.h>
#include <vector>
#include <iostream>

extern "C" {
#include "vl/generic.h"
#include "vl/sift.h"
#include "vl/kdtree.h"
#include "vl/random.h"
#include "vl/kmeans.h"
#include "vl/host.h"
}

using namespace std;

vl_size vl_kdforest_query_with_array_copy (VlKDForest * self, vl_uint32 * indexes, vl_size numNeighbors, vl_size numQueries, void * distances, void const * queries);

#endif /* defined(__vlfeat_bow_with_reranking__vl_tools__) */
