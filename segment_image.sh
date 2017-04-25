#!/bin/bash


rm -f OBJECT BACKGROUND GRAPH MASK *.out
echo "Compiling code."
export LD_LIBRARY_PATH=./devil/lib
echo "Getting constraints from user."
python constraint_adder.py $1
echo "Reducing image to a graph."
python getWeights.py $1 OBJECT BACKGROUND > GRAPH
echo "Finding min-cut for reduced image graph."
make && ./a.out < GRAPH > MASK
echo "Visualizing segmented image."
python visualizer.py $1 MASK

echo "Segmented image saved."
# rm -f OBJECT BACKGROUND GRAPH MASK *.out