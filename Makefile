all:
	g++ pushRelabelCPU.cpp -o pushRelabel.out

clean:
	rm -f *.out graph OBJECT BACKGROUND GRAPH MASK
