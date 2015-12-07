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
#include "vl_tools.h"
#include "test_extract.h"

#include "armadillo"

using namespace cv;


int main(int argc, const char * argv[]) {
    
    test_extract();
    
    // 读取SIFT描述子文件名, imgsList_in_TXT保存的是.siftb文件名
    string imgsList_in_TXT = "/Users/willard/codes/cpp/opencv-computer-vision/vlfeat-bow-with-reranking/vlfeat-bow-with-reranking/imageNamesList.txt";
    vector<string> imgsFullPath;
    string dirName;
    ifstream file(imgsList_in_TXT);
    std::string temp;
    while(std::getline(file, temp)) {
        cout << temp << endl;
        /*const size_t found = temp.find_last_of("/\\");
         dirName = temp.substr(0,found) + "/";
         vocabularyFiles.push_back(temp.substr(found+1));*/
        imgsFullPath.push_back(temp);
    }
    long int imgsNum = imgsFullPath.size();
    uint sift_desc_dim = 128;
    uint sift_keypoint_dim = 4;
    
    // 测试匹配效果
    string descriptor_Query_path = imgsFullPath[61];
    string descriptor_Object_path = imgsFullPath[63];
    FeatureSet* fQuerySet = readSIFTFile(descriptor_Query_path, sift_keypoint_dim, sift_desc_dim);
    FeatureSet* fObjectSet = readSIFTFile(descriptor_Object_path, sift_keypoint_dim, sift_desc_dim);
    //fQuerySet->print();
    
    // 创建kd树，做近似最近邻搜索，查询图像
    float* data = new float[(*fQuerySet).m_vDescriptors.size()*sift_desc_dim];
    for (unsigned int i = 0; i < (*fQuerySet).m_vDescriptors.size(); i++){
        for (unsigned int j = 0; j < sift_desc_dim; j++){
            data[j+sift_desc_dim*i] = (*fQuerySet).m_vDescriptors[i][j];
        }
    }
    
    // 查询图片数据检测，仅仅用来验证是否读写正确，验证通过
    for (unsigned int i = 0; i < sift_desc_dim; i++){
        //cout << data[((*fQuerySet).m_vDescriptors.size() - 0)*sift_desc_dim + i] << "\t";
    }
    //cout << endl;
    //cout << "溢出检测： " << data[(*fQuerySet).m_vDescriptors.size()*sift_desc_dim] << "\t" << data[(*fQuerySet).m_vDescriptors.size()*sift_desc_dim + 1] << endl;
    
    VlKDForest* forest = vl_kdforest_new(VL_TYPE_FLOAT, sift_desc_dim, 3, VlDistanceL2);  //创建kd树对象，128维，3棵
    forest->thresholdingMethod = VL_KDTREE_MEDIAN;
    // see http://www.vlfeat.org/api/kdtree_8h.html#ac886f1fd6024a74e9e4a5d7566b2125f for detail about vl_kdforest_build
    vl_kdforest_build(forest, (*fQuerySet).m_vDescriptors.size(), data); // 第2个参数为数据点个数
    VlKDForestSearcher* searcher = vl_kdforest_new_searcher(forest);
    
    // 目标图像
    float* fObjectData = new float[sift_desc_dim*(*fObjectSet).m_vDescriptors.size()];
    for (unsigned int i = 0; i < (*fObjectSet).m_vDescriptors.size(); i++)
        for (unsigned int j = 0; j < sift_desc_dim; j++)
            fObjectData[j+sift_desc_dim*i] = (*fObjectSet).m_vDescriptors[i][j];
    //fObjectSet->print();
    // 数据检测，仅仅用来验证是否读写正确，验证通过
    for (unsigned int i = 0; i < sift_desc_dim; i++){
        cout << fObjectData[((*fObjectSet).m_vDescriptors.size() - 1)*sift_desc_dim + i] << "\t";
    }
    cout << endl;
    cout << "目标图片溢出检测： " << fObjectData[(*fObjectSet).m_vDescriptors.size()*sift_desc_dim] << "\t" << fObjectData[(*fObjectSet).m_vDescriptors.size()*sift_desc_dim + 1] << endl;
    
    // 做最近查找 Searcher object
    vl_size neiborNum = 2;
    VlKDForestNeighbor neighbours[2];
    vl_size fObjectSetNum = (*fObjectSet).m_vDescriptors.size();
    cout << "目标图片SIFT特征点数目： " << fObjectSetNum << endl;
    vl_uint32* index1 = (vl_uint32 *)vl_malloc(sizeof(vl_uint32)*neiborNum*fObjectSetNum);
    float* dist = (float *)vl_malloc(sizeof(float)*neiborNum*fObjectSetNum);
    /* Query the first ten points for now */
    vector<pair<int, int>> matched_index;
    vector<Point2f> queryLoc, objectLoc;
    
    // 测试并行结果是否准确
    vl_kdforest_query_with_array_copy(forest, index1, neiborNum, fObjectSetNum, dist, fObjectData);
    cout << "最近邻：" << index1[0] << " 距离：" << dist[0] << " " "次近邻：" << index1[1] << " 距离：" << dist[1] << endl;
    cout << "最近邻：" << index1[2] << " 距离：" << dist[2] << " " "次近邻：" << index1[3] << " 距离：" << dist[3] << endl;
    cout << "最近邻：" << index1[4] << " 距离：" << dist[4] << " " "次近邻：" << index1[5] << " 距离：" << dist[5] << endl;
    cout << "最近邻：" << index1[neiborNum*fObjectSetNum-2] << " 距离：" << dist[neiborNum*fObjectSetNum-2] << " " "次近邻：" << index1[neiborNum*fObjectSetNum-1] << " 距离：" << dist[neiborNum*fObjectSetNum-1] << endl;
    cout << "==================================================" << endl;
    
    for(int i=0; i < (*fObjectSet).m_vDescriptors.size(); i++){
        //cout << fObjectData + 128*i << endl;
        //cout << *((float const *)fObjectData + 128*i) << endl;
        //cout << *((float const *)fObjectData + 128*i+1) << endl;
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
    
    //显示1nn<0.8*2nn匹配的点对
    string imgfn = std::string(descriptor_Query_path.begin(), descriptor_Query_path.end()-5) + "jpg";
    string objFileName = std::string(descriptor_Object_path.begin(), descriptor_Object_path.end() - 5) +"jpg";
    
    cv::Mat srcColorImage = cv::imread(imgfn);
    cv::Mat dstColorImage = cv::imread(objFileName);
    //drawMatch(imgfn, objFileName, queryLoc, objectLoc);
    plotMatches(srcColorImage, dstColorImage, queryLoc, objectLoc);
    
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
    plotMatches(srcColorImage, dstColorImage, queryInliers, sceneInliers);
    //fQuerySet->print();
    // Clean up
    if (fQuerySet) {
        delete fQuerySet;
    }
    
    //===============================BOW=======================================
    // 获取所有SIFT的特征点数目, 已确认没问题
    vector<unsigned int> descrNums;
    long int siftNum = 0;
    for(unsigned int i = 0; i < imgsNum; ++i){
        FeatureSet* singleSet = readSIFTFile(imgsFullPath[i], sift_keypoint_dim, sift_desc_dim);
        descrNums.push_back((unsigned int)(*singleSet).m_vDescriptors.size());
        siftNum += (*singleSet).m_vDescriptors.size();
        cout << "第 " << i << "幅图像描述子数目： " << (*singleSet).m_vDescriptors.size() << endl;
    }
    cout << "所有图像总的SIFT数目： " << siftNum << endl;
    
    // 将所有SIFT描述子装入一维数组中，通过检查
    double* dataSIFT = (double *)vl_malloc(sizeof(double)*sift_desc_dim*siftNum);
    static long int tmpNum = 0;
    for(unsigned int i = 0; i < imgsNum; ++i){
        FeatureSet* fSet = readSIFTFile(imgsFullPath[i], sift_keypoint_dim, sift_desc_dim);
        for (unsigned int j = 0; j < (*fSet).m_vDescriptors.size(); j++)
            for (unsigned int k = 0; k < 128; ++k){
                //cout << k+128*j + tmpNum << endl;
                dataSIFT[k+128*j + tmpNum] = (*fSet).m_vDescriptors[j][k];
            }
        tmpNum += (*fSet).m_vDescriptors.size()*128;
        //cout << tmpNum << endl;
    }
    
    cout << "总的sift描述子元素： " << tmpNum << endl;
    
    // 测试最后一幅图像的描述子是否写入准确写入数组里面
    cout << "#################未写入之前的描述子######################" << endl;
    FeatureSet* fSetTmpp = readSIFTFile(imgsFullPath[imgsNum - 1], sift_keypoint_dim, sift_desc_dim);
    for(int i = 0; i < 128; ++i)
        cout << (*fSetTmpp).m_vDescriptors[783][i] << '\t';
    cout << endl;
    cout << "##################写入之后的描述子######################" << endl;
    for (unsigned int i = 0; i < sift_desc_dim; i++){
        cout << dataSIFT[tmpNum - 128 + i] << "\t";
    }
    cout << endl;
    cout << "数组越界测试： " << dataSIFT[tmpNum] << '\t' << dataSIFT[tmpNum + 1] << "\t" << dataSIFT[tmpNum + 2] << endl;
    
    // 进行kmeans聚类
    vl_size centersNum = 5000;
    vl_size maxiter = 10;
    vl_size maxComp = 100;
    vl_size maxrep = 10;
    vl_size ntrees = 8;
    VlKMeansAlgorithm algorithm = VlKMeansANN ;
    //VlKMeansAlgorithm algorithm = VlKMeansLloyd ;
    //VlKMeansAlgorithm algorithm = VlKMeansElkan ;
    VlVectorComparisonType distance = VlDistanceL2 ;
    VlKMeans *km = vl_kmeans_new(VL_TYPE_DOUBLE, distance);  // 初始化
    vl_kmeans_set_verbosity	(km,1);  // 显示聚类每次迭代的信息
    vl_kmeans_set_max_num_iterations (km, maxiter) ;
    vl_kmeans_set_max_num_comparisons (km, maxComp) ;
    vl_kmeans_set_num_repetitions (km, maxrep) ;
    vl_kmeans_set_num_trees (km, ntrees);
    vl_kmeans_set_algorithm (km, algorithm);
    vl_kmeans_cluster(km, dataSIFT, sift_desc_dim, siftNum, centersNum);
    float *centers = (float *)vl_malloc(sizeof(float)*sift_desc_dim*centersNum);
    centers = (float *)vl_kmeans_get_centers(km); // 获取类中心
    
    cout << "======================聚类中心=========================" << endl;
    for(int i = 12800 - 128; i < 128000; ++i)
        cout << centers[i] << "," << '\t';
    cout << endl;
    
    cout << "验证聚类中心数目： " << vl_kmeans_get_num_centers(km) << endl;
    cout << "数组越界测试： " << centers[128000] << '\t' << centers[128001] << endl;
    vl_kmeans_delete(km);
    
    // 用聚类中心创建kd树
    VlKDForest* centersForest = vl_kdforest_new(VL_TYPE_FLOAT, sift_desc_dim, ntrees, VlDistanceL2);  //创建kd树对象，128维，8棵
    centersForest->thresholdingMethod = VL_KDTREE_MEDIAN;
    vl_kdforest_build(centersForest, centersNum, centers);
    
    // 做最近查找 Searcher object
    vl_size nearestNum = 1;
    vl_uint32* words = (vl_uint32 *)vl_malloc(sizeof(vl_uint32)*nearestNum*siftNum);
    float* wordDists = (float *)vl_malloc(sizeof(float)*nearestNum*siftNum);
    vl_kdforest_query_with_array_copy(centersForest, words, nearestNum, siftNum, wordDists, dataSIFT);
    
    // 测试查看聚类中心
    /*for(int i = 0; i < 100; ++i)
        cout << words[i] << "," << '\t';
    cout << endl;*/
    
    
    // 生成单词直方图
    cout << "图像数目： " << descrNums.size() << endl;
    arma::sp_mat Hist(descrNums.size(), centersNum);
    static long int count = 0;
    for (unsigned int i = 0; i < descrNums.size(); i++){
        unsigned int* desrcElementsTmp = new unsigned int[descrNums[i]];
        memcpy(desrcElementsTmp, words + count, descrNums[i] * sizeof(words[0]));
        //cout << desrcElementsTmp[0] << '\t' << desrcElementsTmp[1] << '\t' << desrcElementsTmp[2] << '\t' << desrcElementsTmp[3] << '\t' << desrcElementsTmp[4] << '\t' <<endl;
        //cout << desrcElementsTmp[5] << '\t' << desrcElementsTmp[6] << '\t' << desrcElementsTmp[7] << '\t' << desrcElementsTmp[8] << '\t' << desrcElementsTmp[9] << '\t' << desrcElementsTmp[10] << '\t' <<endl;
        //cout << endl;
        arma::sp_mat X(1, centersNum);
        X.zeros();
        for (unsigned int j = 0; j < descrNums[i]; j++){
            //cout << desrcElementsTmp[j] << "\t" << endl;
            X(0, desrcElementsTmp[j]) += 1;
        }
        Hist.row(i) = X;
        count += descrNums[i];
        delete desrcElementsTmp;
    }
    
    // 计算逆文档词频
    arma::mat idf(1, centersNum);
    for(unsigned int i = 0; i < centersNum; i++){
        cout << (Hist.col(i)).n_nonzero << endl;
        double everyIEF;
        if((Hist.col(i)).n_nonzero != 0)
           everyIEF = log(imgsNum/(Hist.col(i)).n_nonzero);
        idf(0, i) = everyIEF;
    }
    
    idf.print("idf:");
    
    // tf*idf
    for(unsigned int i = 0; i < imgsNum; i++){
        Hist.row(i) = Hist.row(i)%idf; //%元素乘
        if(norm(Hist.row(i), 2) != 0)
            Hist.row(i) = Hist.row(i)/norm(Hist.row(i), 2); // norm(X, 2), 2表示2范数
    }
    
    // 计算余弦距离
    //Hist.print("Hist:");
    vector<float> dotProducts;
    arma::sp_mat queryVec = Hist.row(0);
    for (unsigned int i = 0; i < imgsNum; i++){
        double tmp = arma::dot(queryVec, Hist.row(i));
        dotProducts.push_back(tmp);
        cout << tmp << "\t";
    }
    
    return EXIT_SUCCESS;
}