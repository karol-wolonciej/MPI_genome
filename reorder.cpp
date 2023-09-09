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

#include <bits/stdc++.h>

#define root 0

using namespace std;

void prepareDataForReorderSent(vector<int64>* __restrict__ B, 
                               vector<int64>* __restrict__ SA, 
                               int64 newNodeSize, 
                               int64 nodeSize,
                               int64 dataSize,
                               int64 h,
                               vector<vector<TwoInts64>>* __restrict__ dataForPartitions,
                               int rank,
                               int worldSize) {
    TwoInts64 data;
    int nodeToSend;

    for (int i = 0; i < nodeSize; i++) {
        nodeToSend = getNodeToSend(SA->data()[i], newNodeSize);
        data.i1 = SA->data()[i];
        data.i2 = B->data()[i];
        dataForPartitions->data()[nodeToSend].push_back(data);
    }
}


void reorder_and_rebalance(vector<int64>* __restrict__ B, 
                           vector<int64>* __restrict__ B_new, 
                           vector<int64>* __restrict__ SA,
                           vector<int64>* __restrict__ SA_second_pointer,
                           HelpingVectorsSendingOperations* __restrict__ helpVectors,
                           int rank, 
                           int worldSize) {

    do_sending_operation(B, 
                         B_new, 
                         SA,
                         SA_second_pointer,
                         helpVectors,
                         EMPTY_HELP_PARAM,
                         true, 
                         rank, 
                         worldSize,
                         &prepareDataForReorderSent);
}



