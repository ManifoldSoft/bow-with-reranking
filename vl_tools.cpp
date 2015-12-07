// 模块说明：从VLfeat中到处的模块，方面自行修改定制
// 日期: 2015年9月5号
// 版本： v 0.0.1

#include "vl_tools.h"

// 函数功能：多线程kd树
vl_size vl_kdforest_query_with_array_copy (VlKDForest * self, vl_uint32 * indexes, vl_size numNeighbors, vl_size numQueries, void * distances, void const * queries){
    vl_size numComparisons = 0;
    vl_type dataType = vl_kdforest_get_data_type(self) ;
    vl_size dimension = vl_kdforest_get_data_dimension(self) ;
    
#ifdef _OPENMP
#pragma omp parallel default(shared) num_threads(vl_get_max_threads())
#endif
    {
        vl_index qi ;
        vl_size thisNumComparisons = 0 ;
        VlKDForestSearcher * searcher ;
        VlKDForestNeighbor * neighbors ;
        
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            searcher = vl_kdforest_new_searcher(self) ;
            neighbors = (VlKDForestNeighbor *)vl_calloc (sizeof(VlKDForestNeighbor), numNeighbors) ;
        }
        
#ifdef _OPENMP
#pragma omp for
#endif
        for(qi = 0 ; qi < (signed)numQueries; ++ qi) {
            switch (dataType) {
                case VL_TYPE_FLOAT: {
                    vl_size ni;
                    //cout << numNeighbors << endl;
                    //cout << *((float const *)queries) << endl;
                    //cout << *((float const *)queries+1) << endl;
                    thisNumComparisons += vl_kdforestsearcher_query (searcher, neighbors, numNeighbors, (float const *) (queries) + qi * dimension) ;
                    for (ni = 0 ; ni < numNeighbors ; ++ni) {
                        indexes [qi*numNeighbors + ni] = (vl_uint32) neighbors[ni].index ;
                        if (distances){
                            *((float*)distances + qi*numNeighbors + ni) = neighbors[ni].distance ;
                        }
                    }
                    break ;
                }
                case VL_TYPE_DOUBLE: {
                    vl_size ni;
                    thisNumComparisons += vl_kdforestsearcher_query (searcher, neighbors, numNeighbors,
                                                                     (double const *) (queries) + qi * dimension) ;
                    for (ni = 0 ; ni < numNeighbors ; ++ni) {
                        indexes [qi*numNeighbors + ni] = (vl_uint32) neighbors[ni].index ;
                        if (distances){
                            *((double*)distances + qi*numNeighbors + ni) = neighbors[ni].distance ;
                        }
                    }
                    break ;
                }
                default:
                    abort() ;
            }
        }
        
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            numComparisons += thisNumComparisons ;
            vl_kdforestsearcher_delete (searcher) ;
            vl_free (neighbors) ;
        }
    }
    return numComparisons ;
}
