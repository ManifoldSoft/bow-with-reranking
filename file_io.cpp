//
//  file_io.cpp
//  vlfeat-bow-with-reranking
//
//  Created by willard on 8/24/15.
//  Copyright (c) 2015 wilard. All rights reserved.
//

#include "file_io.h"

FeatureSet* readSIFTFile(string sFilename, uint nFrameLength, uint nDescLength)
{
    FeatureSet* pSet = NULL;
    FILE* pFile = fopen(sFilename.c_str(), "rb");
    
    if (pFile != NULL)
    {
        // Create new feature set
        pSet = new FeatureSet(nDescLength, nFrameLength);
        
        // Read number of features
        int nNumFeatures = 0;
        int nRead = fread(&nNumFeatures, sizeof(int), 1, pFile);
        
        // Safety switch to prevent memory overload
        nNumFeatures = std::min(MAX_NUM_FEATURES, nNumFeatures);
        
        // Read frames and descriptors
        for (int nFeature = 0; nFeature < nNumFeatures; nFeature++)
        {
            float* pFrame = new float[nFrameLength];
            float* pDesc = new float[nDescLength];
            nRead = fread(pFrame, sizeof(float), nFrameLength, pFile);
            nRead = fread(pDesc, sizeof(float), nDescLength, pFile);
            
            float fL2Norm = 0;
            for (uint nDim = 0; nDim < nDescLength; nDim++)
            {
                pDesc[nDim] /= DESC_DIVISOR; // Division is necessary to compensate for multiplier in vl_sift
                fL2Norm += pDesc[nDim] * pDesc[nDim];
            } // nDim
            
            if (isnan(fL2Norm)) {
                delete [] pFrame;
                delete [] pDesc;
                continue;
            }
            pSet->addFeature(pDesc, pFrame);
        } // nFeature
        
        // Close file
        fclose(pFile);
    }
    else
    {
        fprintf(stderr, "Cannot open: %s\n", sFilename.c_str());
        exit(1);
    }
    return pSet;
}


