all:
	g++ pushRelabelCPU.cpp -o pushRelabel.out -g

clean:
	rm -f *.out graph OBJECT BACKGROUND GRAPH MASK
