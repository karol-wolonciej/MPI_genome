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



void rebucket_assign_2h_group_rank(vector<Tuple3>* __restrict__ tuple, 
                                   vector<int64>* __restrict__ B,
                                   vector<int64>* __restrict__ SA, 
                                   bool* __restrict__ allSingletones,
                                   int rank,
                                   int worldSize) {

    int64 indexOffset;
    Tuple3 lastTuple;
    lastTuple.B = -1;
    lastTuple.B2 = -1;
    lastTuple.i = -1;

    if (rank == 0) {
        bool localAllSingletones = true;
        int64 size = tuple->size();
        int64 lastPossibleIndex = size-1;

        if (size > 0) {
            B->resize(size);
            SA->resize(size);
            B->data()[0] = 0;
            SA->data()[0] = tuple->data()[0].i;
            lastTuple.i = 0;

            for (int64 i = 1; i < size; i++) {
                if (tuple3Equal(tuple->data()[i-1], tuple->data()[i])) {
                        localAllSingletones = false;
                        B->data()[i] = B->data()[i-1];
                    }
                else {
                    B->data()[i] = i;
                    lastTuple.i = i;
                }
                SA->data()[i] = tuple->data()[i].i;
            }
        }

        *allSingletones = localAllSingletones;
        indexOffset = lastPossibleIndex + 1;

        lastTuple.B = size > 0 ? tuple->data()[lastPossibleIndex].B : lastTuple.B;
        lastTuple.B2 = size > 0 ? tuple->data()[lastPossibleIndex].B2 : lastTuple.B2;
        
        if (worldSize > 1) {
            MPI_Send(allSingletones, 1, MPI_C_BOOL, 1, root, MPI_COMM_WORLD);
            MPI_Send(&indexOffset, 1, MPI_LONG_LONG_INT, 1, root, MPI_COMM_WORLD);
            MPI_Send(&lastTuple, 1, MPI_Tuple3, 1, root, MPI_COMM_WORLD);
        }
    }
    else {

        MPI_Recv(allSingletones, 1, MPI_C_BOOL,        rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&indexOffset, 1,         MPI_LONG_LONG_INT, rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&lastTuple, 1,           MPI_Tuple3,        rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int64 size = tuple->size();
        int64 lastPossibleIndex = size-1;

        B->resize(size);
        SA->resize(size);

        if (size > 0) {
            if (indexOffset > 0) {
                B->data()[0] = tuple3Equal(lastTuple, tuple->data()[0]) ? lastTuple.i : indexOffset;
                SA->data()[0] = tuple->data()[0].i;
            }
            for (int64 i = (indexOffset == 0 ? 0 : 1); i < size; i++) {

                if (tuple3Equal(tuple->data()[i-1], tuple->data()[i])) {
                        *allSingletones = false;
                        B->data()[i] = B->data()[i-1];
                    }
                else {
                    B->data()[i] = i + indexOffset;
                    lastTuple.i = i + indexOffset;
                }
                SA->data()[i] = tuple->data()[i].i;
            }

            indexOffset += lastPossibleIndex + 1;
            lastTuple.B = size > 0 ? tuple->data()[lastPossibleIndex].B : lastTuple.B;
            lastTuple.B2 = size > 0 ? tuple->data()[lastPossibleIndex].B2 : lastTuple.B2;
        }

        if (rank < worldSize-1) {
            MPI_Send(allSingletones, 1, MPI_C_BOOL, rank+1, rank, MPI_COMM_WORLD);
            MPI_Send(&indexOffset, 1, MPI_LONG_LONG_INT, rank+1, rank, MPI_COMM_WORLD);
            MPI_Send(&lastTuple, 1, MPI_Tuple3, rank+1, rank, MPI_COMM_WORLD);
        }
    }

    MPI_Bcast(allSingletones, 1, MPI_C_BOOL, worldSize-1, MPI_COMM_WORLD);
}



void rebucket_assign_h_group_rank(vector<Tuple2>* __restrict__ tuple, 
                                  vector<int64>* __restrict__ B, 
                                  int rank,
                                  int worldSize) {

    int64 indexOffset;
    Tuple2 lastTuple;
    lastTuple.B[0] = '*';
    lastTuple.i = -1;

    if (rank == 0) {
        int64 size = tuple->size();
        int64 lastPossibleIndex = size-1;

        if (size > 0) {
            B->resize(size);
            B->data()[0] = 0;
            lastTuple.i = 0;

            for (int64 i = 1; i < size; i++) {
                if (tuple2Equal(tuple->data()[i-1], tuple->data()[i])) {
                        B->data()[i] = B->data()[i-1];
                }
                else {
                    B->data()[i] = i;
                    lastTuple.i = i;
                }
            }
        }

        indexOffset = lastPossibleIndex + 1;

        if (size > 0) {
            strncpy(lastTuple.B, tuple->data()[lastPossibleIndex].B, K+1);
        }

        if (worldSize > 1) {
            MPI_Send(&indexOffset, 1, MPI_LONG_LONG_INT, 1, root, MPI_COMM_WORLD);
            MPI_Send(&lastTuple, 1, MPI_Tuple2, 1, root, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(&indexOffset, 1,         MPI_LONG_LONG_INT, rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&lastTuple, 1,           MPI_Tuple2,        rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int64 size = tuple->size();
        int64 lastPossibleIndex = size-1;

        if (size > 0) {
            B->resize(size);
            B->data()[0] = tuple2Equal(lastTuple, tuple->data()[0]) ? lastTuple.i : indexOffset;

            for (int64 i = 1; i < size; i++) {
                if (tuple2Equal(tuple->data()[i-1], tuple->data()[i])) {
                        B->data()[i] = B->data()[i-1];
                }
                else {
                    B->data()[i] = i + indexOffset;
                    lastTuple.i = i + indexOffset;
                }
            }

        }

        indexOffset += lastPossibleIndex + 1;

        if (size > 0) {
            strncpy(lastTuple.B, tuple->data()[lastPossibleIndex].B, K+1);
        }

        if (rank < worldSize-1) {
            MPI_Send(&indexOffset, 1, MPI_LONG_LONG_INT, rank+1, rank, MPI_COMM_WORLD);
            MPI_Send(&lastTuple, 1, MPI_Tuple2, rank+1, rank, MPI_COMM_WORLD);
        }
    }

}


