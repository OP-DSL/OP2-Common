CC = icc
CPP = icpc

all: compare_dat compare_bin

compare_dat:	compare.cpp
		$(CPP) compare.cpp -o compare_dat

compare_bin: comparebin.cpp
		$(CPP) comparebin.cpp -o compare_bin

clean:
	rm -f compare_dat compare_bin
