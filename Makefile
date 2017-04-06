GPPFLAGS = -I./devil/include  --gpu-architecture=compute_35 -L ./devil/lib
LDFLAGS =  -O2 -lm -lstdc++ -lIL -lILU


all:
	g++ pushRelabelCPU.cpp -o pushRelabel.out -g
	nvcc imageFlow.cu ${GPPFLAGS} ${LDFLAGS}

clean:
	rm -f *.out graph OBJECT BACKGROUND GRAPH MASK
