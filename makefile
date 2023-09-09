zadanie2:
	CC -O3 -std=c++2a --std=c++2a zadanie2.cpp data_source.cpp -o genome_index

clean :
	rm -f *.o *.out *.err genome_index
