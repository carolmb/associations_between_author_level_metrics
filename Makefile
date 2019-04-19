all: main

main:
	g++ -std=c++1z -g src/* -I inc/ -I/usr/local/include/igraph -L/usr/local/lib -ligraph -o main.out 
