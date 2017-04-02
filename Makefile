# CPPFLAGS=-Wno-deprecated-declarations
# LDFLAGS= -O2 -lm -lstdc++ -lIL -lILU

CPPFLAGS=-Wno-deprecated-declarations -I./devil/include -std=c++11
LDFLAGS= -O2 -L./devil/lib -lm -lstdc++ -lIL -lILU

all:
	export LD_LIBRARY_PATH=./devil/lib
	g++ -o getWeights.out getWeights.cpp ${CPPFLAGS} ${LDFLAGS}
	g++ pushRelabelCPU.cpp -o pushRelabel.out

clean:
	rm -f *.out graph OBJECT BACKGROUND

