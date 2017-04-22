#!/bin/bash


rm -f OBJECT BACKGROUND GRAPH MASK
echo "Compiling code"
export LD_LIBRARY_PATH=./devil/lib
make
echo "Getting constraints from user"
python constraint_adder.py $1
echo "Reducing image to a graph"
python getWeights.py $1 OBJECT BACKGROUND > GRAPH
echo "Funding min-cut for reduced image graph"
./pushRelabel.out < GRAPH > MASK
echo "Visualizing segmented image"
python visualizer.py $1 MASK

echo "Segmented image stored in images/segmented.png"
rm -f OBJECT BACKGROUND GRAPH MASK