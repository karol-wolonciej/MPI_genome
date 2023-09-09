#include <iostream> 
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>


#include <omp.h>
#include <mpi.h>



#ifndef   auxiliary
#define   auxiliary
    #include "auxiliary.cpp"
#endif

#include<bits/stdc++.h>




using namespace std;


char getRandomCharacter() {
    int v = rand() % 4;
    switch (v) {
    case 0:
        return 'A';
        break;
    case 1:
        return 'C';
        break;
    case 2:
        return 'G';
        break;
    }
    return 'T';
}

void fillCharArray(char* arr) {
    for (int i = 0; i < charArrayLen; i++) {
        arr[i] = getRandomCharacter();
    }
    arr[charArrayLen] = '\0';
}