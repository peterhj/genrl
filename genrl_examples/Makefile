# For building OpenBLAS.
CC := gcc-4.9
CXX := g++-4.9
FC := gfortran-4.9

.PHONY: all clean

all:
	CC=$(CC) CXX=$(CXX) FC=$(FC) cargo build --release

clean:
	cargo clean
