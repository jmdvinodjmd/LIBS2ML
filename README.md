## Work in progress...


LIBS2ML: A Library for Scalable Second Order Machine Learning Algorithms
===========================================================================


## About LIBS2ML
LIBS2ML is a library based on scalable second order learning algorithms for solving large-scale problems, i.e., big data problems in machine learning. LIBS2ML has been developed using MEX files, i.e., C++ with MATLAB/Octave interface to take the advantage of both the worlds, i.e., faster learning using C++ and easy I/O using MATLAB. Most of the available libraries are either in MATLAB/Python/R which are very slow and not suitable for large-scale learning, or are in C/C++ which does not have easy ways to take input and display results. So LIBS2ML is completely unique due to its focus on the scalable second order methods, the hot research topic, and being based on MEX files. Thus it provides researchers a comprehensive environment to evaluate their ideas and it also provides machine learning practitioners an effective tool to deal with the large-scale learning problems. LIBS2ML is an open-source, highly efficient, extensible, scalable, readable, portable and easy to use library. The library can be downloaded from the URL: \url{https://github.com/jmdvinodjmd/LIBS2ML}.


## MANUSCRIPT
You will find the accompanying manuscript by Tuesday, as it is submitted to arXiv and will be online by Tuesday.
+ Vinod Kumar Chauhan, Anuj Sharma, and Kalpana Dahiya. LIBS2ML: A Library for Scalable Second Order Machine Learning Algorithms. arXiv, April 2019. URL http:


## PROBLEMS
Currently, LIBS2ML contains following problems but we will add more problems in the future.
- Logistic Regression (l2-regularized)
- Support Vector Machine (l2-regularized and l2-loss)

Folders and files
---------
<pre>
./                      - This is top directory.
./README.md             - This is readme file.
./run_this_first.m      - The scipt adds folders to path so need to run this first.
./make.m                - This MATLAB/Octave script is used to compile the source code.
|data/                  - Contains the datasets used in the expriments
|demos/                 - MATLAB driver scripts for running experiments are present here.
|methods/               - The learning algorithms are defined here.
|mexfiles/              - It contains MEX files, written in C++, which act as interface between MATLAB/Octave and C++.
|others/                - Auxiliary helpful code is present in this folder.
|problems/              - This contains class definitions for each problem.
</pre>    


## LEARNING METHODS
Currently, LIBS2ML Contains following learning algorithms but we will add more methods in the future.
- TRON (Trust RegiOn Newton) method
    + Chih-Yang Hsia, Wei-Lin Chiang, and Chih-Jen Lin. 2018. Preconditioned Conjugate Gradient Methods in Truncated Newton Frameworks for Large-scale Linear Classification. In Proceedings of the Tenth Asian Conference on Machine Learning (Proceedings of Machine Learning Research). PMLR.
- STRON (Stochastic Trust RegiOn Newton) method
    + Vinod Kumar Chauhan, Anuj Sharma, and Kalpana Dahiya. Stochastic Trust Region Inexact Newton Method for Large-scale Machine Learning. arXiv, dec 2018c. URL http://arxiv.org/abs/1812.10426.
- Newton-CG
    + R. Byrd, G. Chin,W. Neveitt, and J. Nocedal. 2011. On the Use of Stochastic Hessian Information in Optimization Methods for Machine Learning. SIAM Journal on Optimization 21, 3 (2011), 977–995. https://doi.org/10.1137/10079923X
- LBFGS
    + Dong C. Liu and Jorge Nocedal. 1989. On the limited memory BFGS method for large scale optimization. Mathematical Programming 45, 1 (1989), 503–528.
- SQN (Stochasit Quasi-Newton) method
    + Richard H. Byrd, S. L. Hansen, Jorge Nocedal, and Yoram Singer. 2016. A Stochastic Quasi-Newton Method for Large-Scale Optimization. SIAM Journal on Optimization 26, 2 (2016), 1008–1031.
- SVRG-SQN
    + Philipp Moritz, Robert Nishihara, and Michael I. Jordan. 2016. A Linearly-Convergent Stochastic L-BFGS Algorithm. In AISTATS.
- SVRG-LBFGS
    + Ritesh Kolte, Murat Erdogdu, and Ayfer Ozgur. 2015. Accelerating SVRG via second-order information. In NIPS Workshop on Optimization for Machine Learning.
- STRON-PCG
    + Vinod Kumar Chauhan, Anuj Sharma, and Kalpana Dahiya. Stochastic Trust Region Inexact Newton Method for Large-scale Machine Learning. arXiv, dec 2018c. URL http://arxiv.org/abs/1812.10426.
- STRPM-PCG-Prog
    + Vinod Kumar Chauhan, Anuj Sharma, and Kalpana Dahiya. Stochastic Trust Region Inexact Newton Method for Large-scale Machine Learning. arXiv, dec 2018c. URL http://arxiv.org/abs/1812.10426.
- STRON-SVRG
    + Vinod Kumar Chauhan, Anuj Sharma, and Kalpana Dahiya. Stochastic Trust Region Inexact Newton Method for Large-scale Machine Learning. arXiv, dec 2018c. URL http://arxiv.org/abs/1812.10426.


This documentation provides information to the users and developers for using LIBS2ML.


Table of Contents
=================
- Installation
- Quick Start Example
- Extension
- More Information


## Installation
LIBS2ML works on MATLAB/Octave so it is platform independent and easy to install.
Download the latest version of the library from the following URL: https://github.com/jmdvinodjmd/LIBS2ML

1. Prerequisite
    + MATLAB/Octave with compatiable C++ Compiler. 
You can check if your MATLAB/Octave has compatible complier or not using command ``mex -setup C++" on MATLAB/Octave command prompt. If you see some compliers as output that means your ready to go, otherwise (in MATLAB) you will see a link in the output to find the compatiable c++ compliers, so follow that link and install the required compliers.

2. Compile the source files:
    + Run `run_this_first.m' MATLAB script which will add the folders to the path.
    + Run the `make.m' MATLAB script which compiles the source code.

Now the library is ready to use. You can look at ``Quick Start Example" section.



## Quick Start Example
Here we provide an example of using LIBS2ML. For this we use MATLAB script ``exp_run.m" which uses ``mushroom.mat" dataset, which is present in ``data" folder, and sets some values to different parameters which can be tuned depending upon the problem and the solver. This compares the selected solvers for their convergence against time and accuracy.
- First Run `run_this_first.m' MATLAB script, if not done earlier, which will add the folders to the path.
- `exp_run.m', which is MATLAB driver script for running the experiments.
 

## Extension
LIBS2ML is designed in a modular fashion and problem, solver, driver script and I/O are all kept separate. We can easily extend the library by adding more problems, solvers and even can change the input and output (display of results) very easily, with a little change. We can extend the lirary in following ways:

+ Add more problems
    1. For this, you need define a problem class which defines gradient, Hessian and other necessary informatioin.
    2. Include the file in MEX ``interface.cpp" file
    3. Define an enumeration for the problem in ``helpers.h" header file.
    4. Add clauses for the new class in the MEX ``interface.cpp" file, just like other existing classes.
    5. Specify the problem class in the MATLAB driver script.

+ Add more methods
    1. For this, you need define a method class which defines the learning algorithm for the method.
    2. Include the file in MEX ``interface.cpp" file
    3. Define an enumeration for the method in ``helpers.h" header file.
    4. Add clauses for the new method in the MEX ``interface.cpp" file, just like other existing classes.
    5. Specify the method class in the MATLAB driver script.

+ Change I/O:
    1. For changing the output, i.e, graphs you need to change the ``plot_graph.m" MATLAB file.
    2. For changing the input format for the data, you need to change MEX ``interface.cpp" file. But remember, all the methods/problems need to be changed according to the data format.


## More Information

LIBS2ML is an open source library and released under the Apache 2.0 open source license.

Please cite LIBS2ML if you find it useful:

```
@article{libs2ml2019,
  title={LIBS2ML: A Library for Scalable Second Order Machine Learning Algorithms},
  author={Vinod Kumar Chauhan, Anuj Sharma and Kalpana Dahiya},
  journal={arXiv},
  pages = {1--5},
  year={2019}
}
```
You can have a look into the lessons learnt from the development of LIBS2ML from following blog:
[Lessons Learnt from Developing a C++ Library](https://jmdvinodjmd.blogspot.com/2018/12/lessons-learnt-from-developing-c-library.html)

## Contact us
This library is a work in progress so any questions and suggestions are welcomed. Please contact:
[Dr. Anuj Sharma](https://sites.google.com/view/anujsharma/) and [Vinod Kumar Chauhan](https://sites.google.com/site/jmdvinodjmd/), and send email to- anujs **AT** pu **dot** ac **dot** in and cc to jmdvinodjmd **AT** gmail **dot** com.

## Release Notes
- ...
