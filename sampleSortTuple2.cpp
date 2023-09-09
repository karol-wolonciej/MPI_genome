#include <iostream> 
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <cstdio>
#include <vector>
#include <numeric>


#include <omp.h>
#include <mpi.h>



#ifndef   auxiliary
#define   auxiliary
    #include "auxiliary.cpp"
#endif

#include<bits/stdc++.h>

#define root 0


using namespace std;



const int64 binarySearchTuple2(const vector<Tuple2>* __restrict__ arr, 
                               Tuple2 tuple, 
                               int64 l, 
                               int64 r)
{
    if (tuple2Greater(tuple, arr->data()[arr->size()-1])) {
        return arr->size();
    }

    if (tuple2Smaller(tuple, arr->data()[0])) {
        return 0;
    }

    if (r >= l) {
        int64 mid = l + (r - l) / 2;

        if (tuple2Equal(arr->data()[mid], tuple) || (tuple2Smaller(tuple, arr->data()[mid]) && tuple2Greater(tuple, arr->data()[mid-1])))
            return mid;

        if (tuple2Greater(arr->data()[mid], tuple))
            return binarySearchTuple2(arr, tuple, l, mid - 1);
 
        return binarySearchTuple2(arr, tuple, mid + 1, r);
    }
    return -1;
}


void findPivotPositionsTuple2(const vector<Tuple2>* __restrict__ arr, 
                              vector<Tuple2>* __restrict__ pivotsTuples, 
                              vector<int64>* __restrict__ pivotsPositions, 
                              int rank) {

    pivotsPositions->resize(pivotsTuples->size());

    for (int64 i = 0; i < (int64) pivotsTuples->size(); i++) {
        pivotsPositions->data()[i] = binarySearchTuple2(arr, pivotsTuples->data()[i], 0, arr->size());
    }
	pivotsPositions->push_back(arr->size());
}



void getNextPartialPivotsTuple2(vector<Tuple2>* __restrict__ arr, 
                                vector<Tuple2>* __restrict__ partialArr, 
                                vector<int64>* __restrict__ pivotsPosition, 
                                vector<int64>* __restrict__ partialPivotsPosition,
                                vector<int>* __restrict__ scattervPositions,
                                vector<int>* __restrict__ displacement,
                                int worldSize) {

    int64 partialArraSize = 0;
    int64 nextSendSize;

    partialArr->clear();
    
    for (int64 i = 0; i < (int64) pivotsPosition->size(); i++) {
        nextSendSize = getNextSendSize(partialPivotsPosition->data()[i], pivotsPosition->data()[i], worldSize);
        partialArraSize += nextSendSize;
    }

    int64 displacementSum = 0;

    for (int64 i = 0; i < (int64) pivotsPosition->size(); i++) {
        nextSendSize = getNextSendSize(partialPivotsPosition->data()[i], pivotsPosition->data()[i], worldSize);
		scattervPositions->data()[i] = nextSendSize;
        partialArr->insert(partialArr->end(), arr->begin() + partialPivotsPosition->data()[i], arr->begin() + partialPivotsPosition->data()[i] + nextSendSize);
        partialPivotsPosition->data()[i] += nextSendSize;
        displacement->data()[i] = displacementSum;
        displacementSum += nextSendSize;
    }
}



void sendDataToProperPartitionTuple2(vector<Tuple2>* __restrict__ A, 
                                     vector<Tuple2>* __restrict__ A_sampleSorted, 
                                     HelpingVectorsSampleSort2* __restrict__ helpVectors,
                                     int rank, 
                                     int worldSize) {

    A_sampleSorted->clear();

	helpVectors->partialPivotsPosition.resize(helpVectors->pivotsPositions.size());
	helpVectors->scattervPositions.resize(worldSize);
	helpVectors->displacement.resize(worldSize);
    helpVectors->arrivingNumber.resize(worldSize);
    helpVectors->arrivingDisplacement.resize(worldSize);
    
    int64 sizeTmpBuff;

	helpVectors->partialPivotsPosition[0] = 0;
	for (int64 i = 1; i < (int64) helpVectors->pivotsPositions.size(); i++) {
		helpVectors->partialPivotsPosition[i] = helpVectors->pivotsPositions.data()[i-1];
	}

	int numberOfPartSendThisProces = ceil(((double)helpVectors->pivotsPositions.data()[0] / (double)wyslijRaz));
	for (int64 i = 1; i < (int64) helpVectors->pivotsPositions.size(); i++) {
		numberOfPartSendThisProces = max(numberOfPartSendThisProces, (int) ceil((double)((helpVectors->pivotsPositions.data()[i] - helpVectors->pivotsPositions.data()[i-1]) / (double)wyslijRaz)));
	}

    int numberOfLoops;

    MPI_Allreduce(&numberOfPartSendThisProces, &numberOfLoops, 1, MPI_INT, MPI_MAX,MPI_COMM_WORLD);


    for (int partialSends = 0; partialSends < numberOfLoops; partialSends++) {

        getNextPartialPivotsTuple2(A, 
                                   &(helpVectors->partialArr),
                                   &(helpVectors->pivotsPositions), 
                                   &(helpVectors->partialPivotsPosition),
                                   &(helpVectors->scattervPositions),
                                   &(helpVectors->displacement),
                                   worldSize);

        MPI_Alltoall((void*)helpVectors->scattervPositions.data(), 1, MPI_INT, (void*)helpVectors->arrivingNumber.data(), 1, MPI_INT, MPI_COMM_WORLD);


        sizeTmpBuff = accumulate(helpVectors->arrivingNumber.begin(), helpVectors->arrivingNumber.end(), 0);

        helpVectors->arrivingDisplacement.data()[0] = 0;
        for (int64 i = 1; i < (int64) helpVectors->arrivingDisplacement.size(); i++) {
            helpVectors->arrivingDisplacement.data()[i] = helpVectors->arrivingDisplacement.data()[i-1] + helpVectors->arrivingNumber.data()[i-1];
        }
        helpVectors->allArrivingNumbers.insert(helpVectors->allArrivingNumbers.end(), helpVectors->arrivingNumber.begin(), helpVectors->arrivingNumber.end());

        helpVectors->tmp_buff.resize(sizeTmpBuff);

        MPI_Alltoallv(helpVectors->partialArr.data(), 
                      helpVectors->scattervPositions.data(),
                      helpVectors->displacement.data(),
                      MPI_Tuple2,
                      helpVectors->tmp_buff.data(),
                      helpVectors->arrivingNumber.data(),
                      helpVectors->arrivingDisplacement.data(),
                      MPI_Tuple2,
                      MPI_COMM_WORLD);

        A_sampleSorted->insert(A_sampleSorted->end(), helpVectors->tmp_buff.begin(), helpVectors->tmp_buff.end());
        helpVectors->tmp_buff.clear();
    }

    helpVectors->allArrivingDisplacement.resize(helpVectors->allArrivingNumbers.size() + 1);
    helpVectors->allArrivingDisplacement.data()[0] = 0;
    for (int64 i = 1; i < (int64) helpVectors->allArrivingDisplacement.size(); i++) {
        helpVectors->allArrivingDisplacement.data()[i] = helpVectors->allArrivingDisplacement.data()[i-1] + helpVectors->allArrivingNumbers.data()[i-1];
    }
}




void mergeSortedParts(vector<Tuple2>* __restrict__ A, 
                      HelpingVectorsSampleSort2* __restrict__ helpVectors, 
                      int rank) {
                          
    int blocksNumber = helpVectors->allArrivingDisplacement.size()-1;
    int64 roundBlocksNumber = roundToPowerOf2(blocksNumber);
    helpVectors->addPadding.resize(roundBlocksNumber - blocksNumber);
    fill(helpVectors->addPadding.begin(), helpVectors->addPadding.end(), helpVectors->allArrivingDisplacement.data()[blocksNumber]);
    helpVectors->allArrivingDisplacement.insert(helpVectors->allArrivingDisplacement.end(), helpVectors->addPadding.begin(), helpVectors->addPadding.end());
    int blocksNumberWithPadding = helpVectors->allArrivingDisplacement.size()-1;

    for (int mergeStep = 1; mergeStep < blocksNumberWithPadding; mergeStep *= 2)
	{
		int mergesInStep = (blocksNumberWithPadding / (2 * mergeStep));

		for (int i = 0; i < mergesInStep; i++) {
            int64 indexMergeStart = 2 * mergeStep * i;
            int64 indexMergeMid = indexMergeStart + mergeStep;
            int64 indexMergeEnd = indexMergeMid + mergeStep;

			inplace_merge(A->begin() + helpVectors->allArrivingDisplacement.data()[indexMergeStart], 
                          A->begin() + helpVectors->allArrivingDisplacement.data()[indexMergeMid], 
                          A->begin() + helpVectors->allArrivingDisplacement.data()[indexMergeEnd], cmp_tuple2());
		}
	}
}



void sample_sort_MPI_tuple2(vector<Tuple2>* __restrict__ A, 
                            vector<Tuple2>* __restrict__ A_help,
                            HelpingVectorsSampleSort2* __restrict__ helpVectors,
                            int rank, 
                            int worldSize) {

	A_help->clear();
    helpVectors->allArrivingNumbers.clear();

    local_sort_openMP_tuple2(A);

    int64 step = ceil((double) A->size() / (double) worldSize);
    
    int sendNumber = worldSize;
    for (int i = 0; i < worldSize; i++) {
        helpVectors->sample.data()[i] = A->data()[minInt64(i * step, A->size()-1)];   
    }
        
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather((void*)helpVectors->sample.data(), sendNumber, MPI_Tuple2, (void*)helpVectors->rootSampleRecv.data(), sendNumber, MPI_Tuple2, MPI_COMM_WORLD);

    local_sort_openMP_tuple2(&(helpVectors->rootSampleRecv));

    for (int i = 0; i < worldSize-1; i++) {
        helpVectors->broadcastSample.data()[i] = helpVectors->rootSampleRecv.data()[(i+1) * worldSize];
    }

    findPivotPositionsTuple2(A, &(helpVectors->broadcastSample), &(helpVectors->pivotsPositions), rank);
        
	sendDataToProperPartitionTuple2(A, A_help, helpVectors, rank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);

    mergeSortedParts(A_help, helpVectors, rank);
}



