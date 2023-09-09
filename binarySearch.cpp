#include <iostream> 
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cassert>
#include <math.h>  

#include <omp.h>
#include <mpi.h>

#ifndef   auxiliary
#define   auxiliary
    #include "auxiliary.cpp"
#endif





void getPrefixFromGenom(int64 prefixStart,
                        int64 prefixLen,
                        vector<char>* prefix,
                        int* machineWherePrefixStart,
                        int64 dataSize,
                        int64 nodeSize,
                        vector<int64>* machineSizes,
                        vector<char>* nodeCharArray,
                        int rank,
                        int worldSize) {

    int64 querySize = prefixLen;
    int64 machineWherePrefixStartLocal = -1;

    int64 sizeSoFar = 0;
    for (int i = 0; i < worldSize; i++) {
        if (machineSizes->data()[i] + sizeSoFar > prefixStart) {
            machineWherePrefixStartLocal = i;
            break;
        }
        sizeSoFar += machineSizes->data()[i];
    }

    *machineWherePrefixStart = machineWherePrefixStartLocal;

    int64 offset = 0;
    for (int i = 0; i < rank; i++) {
        offset += machineSizes->data()[i];
    }


    int64 lastIndex = prefixStart + querySize + 1;
    int64 sendCount = rank < machineWherePrefixStartLocal ? 0 : minInt64(nodeSize, maxInt64(0, lastIndex - offset));
    if (rank == machineWherePrefixStartLocal) {
        sendCount = minInt64(lastIndex - prefixStart, nodeSize - (prefixStart - offset));
    }

    vector<int64> totalRecvCountFromMachines;
    totalRecvCountFromMachines.resize(worldSize);
    vector<int> recvCountFromMachinesThisRound;
    recvCountFromMachinesThisRound.resize(worldSize);


    vector<int> displs;
    displs.resize(worldSize);

    int64 localMaxPartialSend = ceil((double) sendCount / (double) wyslijRaz);

    int64 globalMaxPartialSend;
    
    MPI_Allreduce(&localMaxPartialSend, &globalMaxPartialSend, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allgather(&sendCount, 
                  1, 
                  MPI_LONG_LONG_INT, 
                  totalRecvCountFromMachines.data(),
                  1, 
                  MPI_LONG_LONG_INT, 
                  MPI_COMM_WORLD);


    int64 sendCountSoFar = 0;
    int64 startCharArrayOffset = rank == machineWherePrefixStartLocal ? prefixStart - offset : 0;
    int64 sendInThisRound;
    int64 totalSendInThisRound;


    vector<vector<char>> prefixParts;
    if (rank == machineWherePrefixStartLocal) {
        prefixParts.resize(worldSize);
    }

    vector<char> receivedPrefixParts;


    
    for (int i = 0; i < globalMaxPartialSend; i++) {

        sendInThisRound = minInt64(wyslijRaz, sendCount - sendCountSoFar);

        MPI_Allgather(&sendInThisRound, 
                      1, 
                      MPI_INT, 
                      recvCountFromMachinesThisRound.data(),
                      1, 
                      MPI_INT, 
                      MPI_COMM_WORLD);

        displs.data()[0] = 0;
        for (int d = 1; d < worldSize; d++) {
            displs.data()[d] = displs.data()[d-1] + recvCountFromMachinesThisRound.data()[d-1];
        }

        totalSendInThisRound = 0;
        MPI_Allreduce(&sendInThisRound, &totalSendInThisRound, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        receivedPrefixParts.resize(totalSendInThisRound);


        MPI_Gatherv(nodeCharArray->data() + sendCountSoFar + startCharArrayOffset,
                    sendInThisRound, 
                    MPI_CHAR, 
                    receivedPrefixParts.data(),
                    recvCountFromMachinesThisRound.data(), 
                    displs.data(), 
                    MPI_CHAR, 
                    machineWherePrefixStartLocal,
                    MPI_COMM_WORLD);



        if (rank == machineWherePrefixStartLocal) {
            for (int m = 0; m < worldSize; m++) {
                prefixParts.data()[m].insert(prefixParts.data()[m].end(), receivedPrefixParts.begin() + displs.data()[m], receivedPrefixParts.begin() + displs.data()[m] + recvCountFromMachinesThisRound.data()[m]);
            }
        }

        sendCountSoFar += sendInThisRound;
    }

    if (machineWherePrefixStartLocal == rank) {
        prefix->clear();
        for (int p = 0; p < worldSize; p++) {
            prefix->insert(prefix->end(), prefixParts.data()[p].begin(), prefixParts.data()[p].end());
        }

    }


}


void getIndex(vector<int64>* machineSizes,
              vector<int64>* machineOffsets,
              int64 index,
              int* containedMachine,
              int rank,
              int worldSize) {
        
    for (int i = 0; i < worldSize; i++) {
        if (machineOffsets->data()[i] <= index && index < machineOffsets->data()[i] + machineSizes->data()[i]) {
            *containedMachine = i;
        }
    }    
}



void findAnyInfixIndexWithPrefixQuery(vector<char>* query, //ma juz \0 na koncu
                                      int64* startIndexWithPrefix,
                                      int64 dataSize,
                                      int64 nodeSize,
                                      vector<int64>* machineSizes,
                                      vector<int64>* SA_machineSizes,
                                      vector<int64>* SA_machineOffsets,
                                      vector<char>* nodeCharArray,
                                      vector<int64>* SA,
                                      int rank,
                                      int worldSize) {
    int64 l, r;
    l = 0;
    r = dataSize-1;
    int64 currIndex = (l + r) / 2;
    int64 SA_index;
    int machineWhereSAstart = -1;
    int64 prefixLen = query->size()-1;
    vector<char> prefix;

    getIndex(SA_machineSizes,
             SA_machineOffsets,
             currIndex,
             &machineWhereSAstart,
             rank,
             worldSize);

    if (rank == machineWhereSAstart) {
        SA_index = SA->data()[currIndex - SA_machineOffsets->data()[machineWhereSAstart]];
    }


    MPI_Bcast(&SA_index, 1, MPI_LONG_LONG_INT, machineWhereSAstart, MPI_COMM_WORLD);


    int machineWherePrefixStart;
    int cmp_dluzsze;
    int cmp_krotsze;

    for (int i = 0; i < dataSize; i++) {
        if (l > r) {
            *startIndexWithPrefix = -1;
            return;
        }


        getPrefixFromGenom(SA_index,
                           prefixLen,
                           &prefix,
                           &machineWherePrefixStart,
                           dataSize,
                           nodeSize,
                           machineSizes,
                           nodeCharArray,
                           rank,
                           worldSize);

        MPI_Barrier(MPI_COMM_WORLD);
        
        if (machineWherePrefixStart == rank) {
            prefix.push_back('\0');
            cmp_dluzsze = strcmp(query->data(), prefix.data());

            prefix.pop_back(); prefix.pop_back(); prefix.push_back('\0');
            cmp_krotsze = strcmp(query->data(), prefix.data());
        }

        MPI_Bcast(&cmp_dluzsze, 1, MPI_INT, machineWherePrefixStart, MPI_COMM_WORLD);
        MPI_Bcast(&cmp_krotsze, 1, MPI_INT, machineWherePrefixStart, MPI_COMM_WORLD);


        
        if (cmp_krotsze == 0 || cmp_dluzsze == 0) {
            *startIndexWithPrefix = currIndex;
            return;
        }



        if (cmp_dluzsze < 0) {
            r = currIndex-1;
        }
        else if (cmp_dluzsze > 0) {
            l = currIndex+1;
        }

        currIndex = (l + r) / 2;

        getIndex(SA_machineSizes,
                 SA_machineOffsets,
                 currIndex,
                 &machineWhereSAstart,
                 rank,
                 worldSize);

        if (rank == machineWhereSAstart) {
            SA_index = SA->data()[currIndex - SA_machineOffsets->data()[machineWhereSAstart]];
        }
        MPI_Bcast(&SA_index, 1, MPI_LONG_LONG_INT, machineWhereSAstart, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    
}





void findMostLeftOrRightPrefix(vector<char>* query, //ma juz \0 na koncu
                               bool leftMost,
                               int64* resultIndex,
                               int64 startIndex,
                               int64 dataSize,
                               int64 nodeSize,
                               vector<int64>* machineSizes,
                               vector<int64>* SA_machineSizes,
                               vector<int64>* SA_machineOffsets,
                               vector<char>* nodeCharArray,
                               vector<int64>* SA,
                               int rank,
                               int worldSize) {


    int64 l, r;
    if (leftMost) {
        l = 0;
        r = startIndex-1;
    }
    else {
        l = startIndex+1;
        r = dataSize-1;
    }

    int64 previousFoundIndex = startIndex;
    int64 currIndex = (l + r) / 2;
    int64 SA_index;
    int machineWhereSAstart = -1;
    int64 prefixLen = query->size()-1;
    vector<char> prefix;

    getIndex(SA_machineSizes,
             SA_machineOffsets,
             currIndex,
             &machineWhereSAstart,
             rank,
             worldSize);

    if (rank == machineWhereSAstart) {
        SA_index = SA->data()[currIndex - SA_machineOffsets->data()[machineWhereSAstart]];
    }


    MPI_Bcast(&SA_index, 1, MPI_LONG_LONG_INT, machineWhereSAstart, MPI_COMM_WORLD);


    int machineWherePrefixStart;
    int cmp_krotsze;

    for (int i = 0; i < dataSize; i++) {
        if (l > r) {
            *resultIndex = previousFoundIndex;
            return;
        }

        getPrefixFromGenom(SA_index,
                           prefixLen,
                           &prefix,
                           &machineWherePrefixStart,
                           dataSize,
                           nodeSize,
                           machineSizes,
                           nodeCharArray,
                           rank,
                           worldSize);

        MPI_Barrier(MPI_COMM_WORLD);
        
        if (machineWherePrefixStart == rank) {
            prefix.pop_back(); prefix.push_back('\0');
            cmp_krotsze = strcmp(query->data(), prefix.data());
        }

        MPI_Bcast(&cmp_krotsze, 1, MPI_INT, machineWherePrefixStart, MPI_COMM_WORLD);
        
        if (cmp_krotsze == 0) {
            previousFoundIndex = currIndex;
            if (leftMost) {
                r = currIndex-1;
            }
            else {
                l = currIndex+1;
            }
        }
        else {
            if (leftMost) {
                l = currIndex+1;
            }
            else {
                r = currIndex-1;
            }
        }

        currIndex = (l + r) / 2;

        getIndex(SA_machineSizes,
                 SA_machineOffsets,
                 currIndex,
                 &machineWhereSAstart,
                 rank,
                 worldSize);

        if (rank == machineWhereSAstart) {
            SA_index = SA->data()[currIndex - SA_machineOffsets->data()[machineWhereSAstart]];
        }
        MPI_Bcast(&SA_index, 1, MPI_LONG_LONG_INT, machineWhereSAstart, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
    }

}





void startEdgePrefixIndexes(vector<char>* query,
                            vector<char>* nodeCharArray,
                            vector<int64>* SA,
                            int64* resultCount,
                            int64 originalNodeSize,
                            int rank,
                            int worldSize) {

    nodeCharArray->resize(originalNodeSize);

    int64 nodeSize = originalNodeSize;
    int64 nodeSAsize = SA->size();
    int64 SA_machineOffset = 0;
    int64 dataSize;

    MPI_Allreduce(&nodeSize, &dataSize, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    vector<int64> machineSizes;
    vector<int64> SA_machineSizes;
    vector<int64> SA_machineOffsets;

    machineSizes.resize(worldSize);
    MPI_Allgather(&nodeSize, 
                  1, 
                  MPI_LONG_LONG_INT, 
                  machineSizes.data(),
                  1, 
                  MPI_LONG_LONG_INT, 
                  MPI_COMM_WORLD);

    SA_machineSizes.resize(worldSize);
    MPI_Allgather(&nodeSAsize, 
                  1, 
                  MPI_LONG_LONG_INT, 
                  SA_machineSizes.data(),
                  1, 
                  MPI_LONG_LONG_INT, 
                  MPI_COMM_WORLD);

    for (int i = 0; i < rank; i++) {
        SA_machineOffset += SA_machineSizes.data()[i];
    }

    SA_machineOffsets.resize(worldSize);
    MPI_Allgather(&SA_machineOffset, 
                  1, 
                  MPI_LONG_LONG_INT, 
                  SA_machineOffsets.data(),
                  1, 
                  MPI_LONG_LONG_INT, 
                  MPI_COMM_WORLD);

    int64 startIndexWithPrefix;

    findAnyInfixIndexWithPrefixQuery(query,
                                     &startIndexWithPrefix,
                                     dataSize,
                                     nodeSize,
                                     &machineSizes,
                                     &SA_machineSizes,
                                     &SA_machineOffsets,
                                     nodeCharArray,
                                     SA,
                                     rank,
                                     worldSize);

    int64 leftMost, rightMost;
    bool isLeftMost = true;
    bool isRightMost = false;

    if (startIndexWithPrefix == -1) {
        *resultCount = 0;
        return;
    }

    findMostLeftOrRightPrefix(query,
                              isLeftMost,
                              &leftMost,
                              startIndexWithPrefix,
                              dataSize,
                              nodeSize,
                              &machineSizes,
                              &SA_machineSizes,
                              &SA_machineOffsets,
                              nodeCharArray,
                              SA,
                              rank,
                              worldSize);


    findMostLeftOrRightPrefix(query,
                              isRightMost,
                              &rightMost,
                              startIndexWithPrefix,
                              dataSize,
                              nodeSize,
                              &machineSizes,
                              &SA_machineSizes,
                              &SA_machineOffsets,
                              nodeCharArray,
                              SA,
                              rank,
                              worldSize);

    int64 result = rightMost - leftMost + 1; 
    *resultCount = result;
}