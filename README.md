This is a work in progress...
===========================================================================
 LIBS2ML: A Library for Scalable Second Order Machine Learning Algorithms
===========================================================================


About LIBS2ML
====================================================
LIBS2ML is a library based on stochastic second order learning algorithms, first of this kind, for solving large-scale problems, i.e., big data problems in machine learning. LIBS2ML has been developed using MEX files, i.e., C++ with MATLAB interface to take the advantage of both the worlds, i.e., faster processing using C++ and easy I/O using MATLAB. Most of the available libraries are either in MATLAB/Python which are very slow and not suitable for large-scale learning, or are in C/C++ which does not have suitable methods for I/O. So LIBS2ML is completely unique due to its focus on the stochastic second order methods, the hot research topic, and being based on MEX files which combines faster processing with easy ways to input and display results as required. Thus it provides researchers a comprehensive environment to evaluate their ideas and it provides machine learning practitioners an effective tool to deal with the large-scale learning problems. LIBS2ML is an open-source, highly efficient, extensible, scalable, readable and portable library which is useful for beginners as well as advanced users and practitioners. The library can be downloaded from the URL: \url{www.google.com}......


LIBS2ML Contains following problems
- Logistic Regression
- SVM

LIBS2ML Contains following learning algorithms
- STRON
- TRON
- 

This documentation provides information to the users and developers for using LIBS2ML.


Table of Contents
=================
- Installation
- Quick Start Example
- Extension
- More Information


Installation
======================
LIBS2ML works on MATLAB/Octave so it is platform independent and easy to install.
Download the latest version of the library from the following URL:
...

1. Prerequisite
    + C++ Compiler compatible with your MATLAB/Octave version. You can verify this using command ``mex -setup C++" on MATLAB.

2. Compile the source files:
    + Run the make.m MATLAB script.

Now the library is ready to use. You can look at ``Quick Start Example" section.


## Known Issues of MEX files


Quick Start Example
===================
Here we provide an example of using LIBS2ML. For this we use MATLAB script ``exp_run.m" which uses ``mushroom.mat" dataset, which is present in ``data" folder, and sets some values to different parameters which can be tuned depending upon the problem and the solver. This compares the selected solvers for their convergence against time and accuracy.
- Run ``exp_run.m", which is MATLAB driver script for running the experiments.
 

Extension
=========
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


More Information
================

LIBS2ML is released under the Apache 2.0 open source license.

Please cite LIBS2ML in your publications if you find it useful:

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
https://jmdvinodjmd.blogspot.com/2018/12/lessons-learnt-from-developing-c-library.html

Contact us
======================
This library is a work in progress so any questions and suggestions are welcomed. Please send your emails to:
anujs@pu.ac.in and cc to jmdvinodjmd@gmail.com

Released On: April, 2019
===================================
The End
