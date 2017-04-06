GPPFLAGS=-I./devil/include
LDFLAGS= -O2 -lm -lstdc++ -lIL -lILU


all:
	g++ pushRelabelCPU.cpp -o pushRelabel.out -g
	nvcc imageFlow.cu ${GPPFLAGS} ${LDFLAGS}

clean:
	rm -f *.out graph OBJECT BACKGROUND GRAPH MASK
