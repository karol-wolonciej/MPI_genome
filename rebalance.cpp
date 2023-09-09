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

#define root 0


using namespace std;





void prepareDataForReorderSent(vector<int64>* __restrict__ A, 
                               int64 newNodeSize, 
                               int64 nodeSize,
                               int64 dataSize,
                               vector<vector<TwoInts64>>* dataForPartitions,
                               int rank,
                               int worldSize) {
    int nodeToSend;
    int64 startIndex = rank * newNodeSize;
    int64 currIndex;

        for (int i = 0; i < nodeSize; i++) {
            currIndex = startIndex + i;
            nodeToSend = getNodeToSend(currIndex, newNodeSize);
            TwoInts64 data;
            data.i1 = currIndex;
            data.i2 = A->data()[i];
            dataForPartitions->data()[nodeToSend].push_back(data);
        }
}


void rebalanceArray(vector<int64>* __restrict__ A, 
                    vector<int64>* __restrict__ A_help,
                    HelpingVectorsSendingOperations* helpVectors,
                    int rank, 
                    int worldSize) {
    int64 dataSize;
    int64 nodeSize = A->size();

    MPI_Allreduce(&nodeSize, &dataSize, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    int64 newNodeSize = ceil(dataSize / (double) worldSize);
    int64 thisNodeNewSize = minInt64(newNodeSize, maxInt64(0, dataSize - rank * newNodeSize));

    A_help->resize(thisNodeNewSize);

    for (int i = 0; i < worldSize; i++) {
        helpVectors->dataForPartitions.data()[i].clear();
    }

    prepareDataForReorderSent(A, newNodeSize, nodeSize, dataSize, &(helpVectors->dataForPartitions), rank, worldSize);

    fill(helpVectors->partialPivotsPosition.begin(), helpVectors->partialPivotsPosition.end(), 0);

    int64 localMaxPartialSend = 0;
    int64 tmpPartialSend = 0;
    for (int i = 0; i < worldSize; i++) {
        tmpPartialSend = ceil((double) helpVectors->dataForPartitions.data()[i].size() / (double) wyslijRaz);
        localMaxPartialSend = maxInt64(localMaxPartialSend, tmpPartialSend);
    }

    int64 globalMaxPartialSend;
    
    MPI_Allreduce(&localMaxPartialSend, &globalMaxPartialSend, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);

    int64 sizeTmpBuff;

    for (int partialSends = 0; partialSends < globalMaxPartialSend; partialSends++) {
        getNextPartialSend(&(helpVectors->dataForPartitions), 
                           &(helpVectors->partialArr), 
                           &(helpVectors->partialPivotsPosition),
						   &(helpVectors->scattervPositions),
						   &(helpVectors->displacement),
                           rank,
                           worldSize);

        MPI_Alltoall((void*)helpVectors->scattervPositions.data(), 1, MPI_INT, (void*)helpVectors->arrivingNumber.data(), 1, MPI_INT, MPI_COMM_WORLD);

        sizeTmpBuff = accumulate(helpVectors->arrivingNumber.begin(), helpVectors->arrivingNumber.end(), 0);

        helpVectors->arrivingDisplacement.data()[0] = 0;
        for (int64 i = 1; i < (int64) helpVectors->arrivingDisplacement.size(); i++) {
            helpVectors->arrivingDisplacement.data()[i] = helpVectors->arrivingDisplacement.data()[i-1] + helpVectors->arrivingNumber.data()[i-1];
        }

        helpVectors->tmp_buff.resize(sizeTmpBuff);

        MPI_Alltoallv(helpVectors->partialArr.data(), 
                helpVectors->scattervPositions.data(),
                helpVectors->displacement.data(),
                MPI_TwoInts64,
                helpVectors->tmp_buff.data(),
                helpVectors->arrivingNumber.data(),
                helpVectors->arrivingDisplacement.data(),
                MPI_TwoInts64,
                MPI_COMM_WORLD);

        int64 offset = rank * newNodeSize;

        for (int64 i = 0; i < (int64) helpVectors->tmp_buff.size(); i ++) {
            A_help->data()[helpVectors->tmp_buff.data()[i].i1 - offset] = helpVectors->tmp_buff.data()[i].i2;
        }

        helpVectors->tmp_buff.clear();
    }
}