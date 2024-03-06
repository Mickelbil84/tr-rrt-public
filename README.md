# Touch & Roll Rapidly Exploring Random Trees (TR-RRT) #

Implementation of our paper, "Tight Motion Planning by Riemannian Optimization for Sliding and Rolling with Finite Number of Contact Points". 

The supplementary material PDF can be found in the root of this repository.
The code can be found in the mrrt module.

## Abstract ##

We address a challenging problem in motion planning where robots must navigate through narrow passages in their configuration space. 
Our novel approach leverages optimization techniques to facilitate sliding and rolling movements across critical regions, 
which represent semi-free configurations, where the robot and the obstacles are in contact. 
Our algorithm seamlessly traverses widely free regions, follows semi-free paths in narrow passages, 
and smoothly transitions between the two types. We specifically focus on scenarios resembling 3D puzzles, 
intentionally designed to be complex for humans by requiring intricate simultaneous translations and rotations. 
Remarkably, these complexities also present computational challenges. Our contributions are threefold: 
Firstly, we solve previously unsolved problems; secondly, we outperform state-of-the-art algorithms on certain problem types; 
and thirdly, we present a rigorous analysis supporting the consistency of the algorithm. 
In the Supplementary Material we provide theoretical foundations for our approach.
This research sheds light on effective approaches to address motion planning difficulties in intricate 3D puzzle-like scenarios.

## Setup ##


1. Initialize pipenv:

        pipenv install
        pipenv shell

2. Install torch, using the instructions in <https://pytorch.org/>

3. (Optional) Run visdom server on a seperate terminal (with `pipenv shell`):

        visdom

4. Before running tests and examples, you should build locally the package (this can be done only once):

        pipenv install -e .


## Usage ##

All scripts can be found under the `scripts` folder.

* To fit an SDF: `scripts/fit.py`
* To run our solver on an instance: `scripts/run_test.py`
* To view the path solution as an animation: `scripts/visualize_path.py`