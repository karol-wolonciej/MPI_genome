#include <iostream> 
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <string>

#include <omp.h>
#include <mpi.h>

#include "SA.cpp"
#include "data_source.h"
#include "addEdgeParts.cpp"
#include "binarySearch.cpp"

using namespace std;



int main(int argc, char** argv) {

	if (argc != 6) {
		cout<<"zla liczba argumentow"<<endl;
		return -1;
	}

	MPI_Init(&argc, &argv);





	int worldRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
	int worldSize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

	int64 n = atoi(argv[1]);
	int64 m = atoi(argv[2]);
	string genome_in(argv[3]);
	string queries_in(argv[4]);
	string queries_out(argv[5]);


	vector<vector<int64>> results;
	vector<vector<char>> queries;

	results.resize(n);
	for (int i = 0; i < n; i++) {
		results.data()[i].resize(m);
	}
	queries.resize(m);


	string line;
	ifstream queriesFile (queries_in);
	if (queriesFile.is_open())
	{
		for (int i = 0; i < m; i++) {
			getline (queriesFile,line);
			queries.data()[i].assign(line.begin(), line.end());
			queries.data()[i].push_back('\0');
		}
		queriesFile.close();
	}


	int64 nodeGenomeSize;
	int64 nodeGenomeOffset;
	int64 originalNodeSize;

	vector<char> nodeCharArray;
	
	// if (worldRank == 0) {
	// 	cout<<n<<endl;
	// 	cout<<m<<endl;
	// 	cout<<genome_in<<endl;
	// 	cout<<queries_in<<endl;
	// 	cout<<queries_out<<endl;
	// }

	//MPI_Tuple3 
	int blockcountTuple3[3]={1,1,1};
    MPI_Aint offsetsTuple3[3] = {offsetof(Tuple3, B), offsetof(Tuple3, B2), offsetof(Tuple3, i)};
    MPI_Datatype dataTypleTuple3[3] = {MPI_LONG_LONG_INT, MPI_LONG_LONG_INT, MPI_LONG_LONG_INT};
    MPI_Type_create_struct(3, blockcountTuple3, offsetsTuple3, dataTypleTuple3, &MPI_Tuple3);
    MPI_Type_commit(&MPI_Tuple3);

	// MPI_Tuple2
	int blockcountTuple2[2]={charArrayLen,1};
    MPI_Aint offsetsTuple2[2] = {offsetof(Tuple2, B), offsetof(Tuple2, i)};
    MPI_Datatype dataTypleTuple2[2] = {MPI_CHAR, MPI_LONG_LONG_INT};
    MPI_Type_create_struct(2, blockcountTuple2, offsetsTuple2, dataTypleTuple2, &MPI_Tuple2);
    MPI_Type_commit(&MPI_Tuple2);

	// TwoInts64
	int blockcountArrData[2]={1,1};
    MPI_Aint offsetsArrData[2] = {offsetof(TwoInts64, i1), offsetof(TwoInts64, i2)};
    MPI_Datatype dataTypeArrData[2] = {MPI_LONG_LONG_INT, MPI_LONG_LONG_INT};
    MPI_Type_create_struct(2, blockcountArrData, offsetsArrData, dataTypeArrData, &MPI_TwoInts64);
    MPI_Type_commit(&MPI_TwoInts64);

	srand (worldRank);

	vector<int64> B_1; 
	vector<int64> B_2; 
	vector<int64> *B_ISA_pointer;

	vector<Tuple2> tuple2_Arr; 
	vector<Tuple2> tuple2_second; 

	vector<Tuple3> tuple3;
	vector<Tuple3> tuple3_second; 

	HelpingVectorsSendingOperations helpVectorsSendingOperations;
	HelpingVectorsSampleSort2 helpVectorsSampleSort2;
	HelpingVectorsSampleSort3 helpVectorsSampleSort3;

	initializeHelpingVectorsSendingOperations(&helpVectorsSendingOperations, worldSize);
	initializeHelpingVectorsSampleSort2(&helpVectorsSampleSort2, worldSize);
	initializeHelpingVectorsSampleSort3(&helpVectorsSampleSort3, worldSize);

	vector<int64> SA, SA_second;

	string inputFile;

	vector<char> addedLastElemsVector; addedLastElemsVector.resize(1);
	fill(addedLastElemsVector.begin(), addedLastElemsVector.end(), '@');
	
	DataSource dataSource((char*) genome_in.c_str());

	int64 resultCount;

	for (int g = 0; g < n; g++) {
		nodeGenomeSize = dataSource.getNodeGenomeSize(g);
		originalNodeSize = nodeGenomeSize;

		nodeGenomeOffset = dataSource.getNodeGenomeOffset(g);
		nodeCharArray.resize(nodeGenomeSize);
		nodeCharArray.insert(nodeCharArray.end(), addedLastElemsVector.begin(), addedLastElemsVector.end());

		dataSource.getNodeGenomeValues(g, nodeCharArray.data());

		while(nodeCharArray.data()[nodeCharArray.size()-1] == '@') {
			nodeCharArray.pop_back();
		}
		
		if (worldRank == worldSize-1) {
			nodeGenomeSize++;
			originalNodeSize++;
			nodeCharArray.data()[nodeGenomeSize-1] = '$';
		}

		addEdgeParts(&nodeCharArray, worldRank, worldSize);

		tuple2_Arr.resize(nodeGenomeSize);
		for (int i = 0; i < nodeGenomeSize; i++) {
			memset(tuple2_Arr.data()[i].B, 0, K+1);
			copy(nodeCharArray.begin() + i, nodeCharArray.begin() + minInt64(i + K, nodeCharArray.size()), tuple2_Arr.data()[i].B);
			tuple2_Arr.data()[i].i = i + nodeGenomeOffset;
		}

		SA_algorithm(&B_1,
					 &B_2,
					 &SA,
					 &SA_second,
					 &tuple2_Arr, 
					 &tuple2_second, 
					 &tuple3, 
					 &tuple3_second,
					 &helpVectorsSendingOperations,
					 &helpVectorsSampleSort2,
					 &helpVectorsSampleSort3,
					 &B_ISA_pointer, 
					 worldRank,
					 worldSize);
		

		for (int q = 0; q < m; q++) {
			startEdgePrefixIndexes(&queries[q],
								   &nodeCharArray,
								   &SA,
								   &resultCount,
								   originalNodeSize,
								   worldRank,
								   worldSize);
			
			results.data()[g].data()[q] = resultCount;
		}
	}




	if (worldRank == root) {
		ofstream resultFile(queries_out);

		for (int q = 0; q < m; q++) {
			for (int g = 0; g < n; g++) {
				resultFile<<results[g][q]; 
				if (g < n-1) {
					resultFile<<" ";
				}
			}
			resultFile<<endl;
		}
		resultFile.close();
	}




	MPI_Finalize();

	return 0;
}


