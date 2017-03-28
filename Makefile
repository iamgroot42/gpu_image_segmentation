# CPPFLAGS=-Wno-deprecated-declarations 
# LDFLAGS= -O2 -lm -lstdc++ -lIL -lILU

CPPFLAGS=-Wno-deprecated-declarations -I./devil/include -std=c++11
LDFLAGS= -O2 -L./devil/lib -lm -lstdc++ -lIL -lILU

all: 
	g++ -g -o getWeights.out getWeights.cpp ${CPPFLAGS} ${LDFLAGS}

clean:
	-rm -f getWeights.out
