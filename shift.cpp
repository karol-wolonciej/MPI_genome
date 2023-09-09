#include <iostream> 
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <cstdio>
#include <cstring>
#include <vector>

#include <omp.h>
#include <mpi.h>



#ifndef   auxiliary
#define   auxiliary
    #include "auxiliary.cpp"
#endif

#include<bits/stdc++.h>


using namespace std;


void prepareDataForShiftSent(vector<int64>* __restrict__ B, 
                             vector<int64>* __restrict__ None1, 
                             int64 newNodeSize, 
                             int64 nodeSize,
                             int64 dataSize,
                             int64 h,
                             vector<vector<TwoInts64>>* __restrict__ dataForPartitions,
                             int rank,
                             int worldSize) {

    int64 curr_i;
    int64 target_i;
    int targetNode;
    int currNode;
    int lastNodeIndex = worldSize-1;
    int data_size_minus_one = dataSize-1;
    TwoInts64 data;
    int64 offset = rank * newNodeSize;
    for (int i = 0; i < (int) B->size(); i++) {
        curr_i = offset + i;
        target_i = curr_i - h;

        currNode = min((int) (curr_i / newNodeSize), lastNodeIndex);
        targetNode = min((int) (target_i / newNodeSize), lastNodeIndex);

        if (target_i >= 0) {
            data.i1 = target_i;
            data.i2 = B->data()[i];
            {
                dataForPartitions->data()[targetNode].push_back(data);
            }
        }
        if (curr_i + h > data_size_minus_one) {
            data.i1 = curr_i;
            data.i2 = 0;
            {
                dataForPartitions->data()[currNode].push_back(data);
            }
        }
    }
}


void shift_by_h(vector<int64>** B, 
                vector<int64>** B_new, 
                vector<int64>* __restrict__ SA,
                vector<int64>* __restrict__ SA_second_pointer,
                HelpingVectorsSendingOperations* helpVectors,
                int64 h,
                int rank, 
                int worldSize) {

    do_sending_operation(*B, 
                         *B_new, 
                         SA,
                         NULL,
                         helpVectors,
                         h, 
                         false,
                         rank, 
                         worldSize,
                         &prepareDataForShiftSent);

}