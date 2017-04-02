# gpu_project
GPU course project


### Running it
* `python constraint_adder.py` to add background and object constraints for an image
* `python getWeights.py <image_path> OBJECT BACKGROUND > GRAPH` to generate weights for the image segmentation graph
* `make` to comiple all relevant files
* `./pr_adj.out < GRAPH` to find the min-cut (and thus segmentation) for the given graph
