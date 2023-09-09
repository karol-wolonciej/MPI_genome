#include <iostream> 
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

#include <omp.h>
#include <mpi.h>

#include "sampleSortTuple2.cpp"
#include "sampleSortTuple3.cpp"
#include "rebucketing.cpp"
#include "reorder.cpp"
#include "shift.cpp"
#include "rebalance.cpp"

#include "generateTestData.cpp"

#ifndef   auxiliary
#define   auxiliary
    #include "auxiliary.cpp"
#endif


using namespace std;



void SA_algorithm(vector<int64>* B_pointer,
                  vector<int64>* B_second_pointer,
                  vector<int64>* SA_pointer,
                  vector<int64>* SA_second_pointer,
                  vector<Tuple2>* tuple2_pointer,
                  vector<Tuple2>* tuple2_second_pointer,
                  vector<Tuple3>* tuple3_pointer,
                  vector<Tuple3>* tuple3_second_pointer,
                  HelpingVectorsSendingOperations *helpVectorsSendingOperations,
                  HelpingVectorsSampleSort2 *helpVectorsSampleSort2,
                  HelpingVectorsSampleSort3 *helpVectorsSampleSort3,
                  vector<int64> **B_ISA_pointer,
                  int worldRank,
                  int worldSize)
    {
	
    MPI_Barrier(MPI_COMM_WORLD);
	sample_sort_MPI_tuple2(tuple2_pointer,
					tuple2_second_pointer,
					helpVectorsSampleSort2,
					worldRank, 
					worldSize);
	switchPointersTuple2(&tuple2_pointer, &tuple2_second_pointer);

	rebucket_assign_h_group_rank(tuple2_pointer, 
					B_pointer, 
					worldRank,
					worldSize);
	initialize_SA(SA_pointer, tuple2_pointer);

	bool done = false;
	for (int64 h = k; true; h*=2) {

		MPI_Barrier(MPI_COMM_WORLD);


		reorder_and_rebalance(B_pointer, 
							  B_second_pointer, 
							  SA_pointer,
							  SA_second_pointer,
							  helpVectorsSendingOperations,
							  worldRank, 
							  worldSize);
		switchPointersInt64(&B_pointer, &B_second_pointer);

		if (done) {
			*B_ISA_pointer = B_pointer;
			break;
		}

		shift_by_h(&B_pointer, 
				   &B_second_pointer, 
				   SA_pointer,
				   NULL,
				   helpVectorsSendingOperations,
				   h,
				   worldRank, 
				   worldSize);

		switchPointersInt64(&SA_pointer, &SA_second_pointer);

		rebalanceArray(SA_pointer, 
                       SA_second_pointer,
                       helpVectorsSendingOperations,
                       worldRank, 
                       worldSize);
		switchPointersInt64(&SA_pointer, &SA_second_pointer);
		
		fillTuple3(B_pointer, 
				   B_second_pointer, 
				   SA_pointer, 
				   tuple3_pointer);

		sample_sort_MPI_tuple3(tuple3_pointer, 
                               tuple3_second_pointer,
                               helpVectorsSampleSort3,
                               worldRank, 
                               worldSize);
		switchPointersTuple3(&tuple3_pointer, &tuple3_second_pointer);
		
		rebucket_assign_2h_group_rank(tuple3_pointer, 
									  B_pointer,
									  SA_pointer,
									  &done,
									  worldRank,
									  worldSize);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}



