# gpu_project
GPU course project


### Running it

* `export LD_LIBRARY_PATH=./devil/lib` to set path for DevIL
* `python constraint_adder.py` to add background and object constraints for an image
* `make` to comiple all relevant files
* `./getWeights.out <image_path> OBJECT BACKGROUND > graph` to generate weights for the image segmentation graph
* `./pr_adj.out < graph` to find the min-cut (and thus segmentation) for the given graph