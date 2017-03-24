# CPPFLAGS=-Wno-deprecated-declarations 
# LDFLAGS= -O2 -lm -lstdc++ -lIL -lILU

CPPFLAGS=-Wno-deprecated-declarations -I./devil/include
LDFLAGS= -O2 -L./devil/lib -lm -lstdc++ -lIL -lILU

all: 
	g++ -o getWeights.out getWeights.cpp ${CPPFLAGS} ${LDFLAGS}

clean:
	-rm -f getWeights.out
