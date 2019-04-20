/**
 *
 * Usage: This file defines the sqn class for the SQN, SVRG_LBFGS and SVRG_SQN in C++.
﻿﻿@article{SQN,
	author    = {Richard H. Byrd and S. L. Hansen and Jorge Nocedal and	Yoram Singer},
	title     = {A Stochastic Quasi-Newton Method for Large-Scale Optimization},
	journal   = {{SIAM} Journal on Optimization},
	volume    = {26},
	number    = {2},
	pages     = {1008-1031},
	year      = {2016}
}
@inproceedings{SVRG_SQN,
	title={A Linearly-Convergent Stochastic L-BFGS Algorithm},
	author={Philipp Moritz and Robert Nishihara and Michael I. Jordan},
	booktitle={AISTATS},
	year={2016}
}
﻿@inproceedings{SVRG_LBFGS,
	title={Accelerating SVRG via second-order information},
	author={Kolte, Ritesh and Erdogdu, Murat and Ozgur, Ayfer},
	booktitle={NIPS Workshop on Optimization for Machine Learning},
	year={2015}
}

 *
 * Input:
 *          problem: oject of the problem class to be solved.
 *          options: it contains different parameter values required for the learning algorithm.
 *          w: parameter vector (updated in place)
 *          output: an instance of output structure (updated in place).
 * Output:
 *          Nothing retured directly but w and output, from the input, are updated in place.
 *
 * Created by V. K. Chauhan on Sept. 17, 2018
 * Modified by V. K. Chauhan on Apr. 16, 2019
 *
**/
#ifndef SQN_H
#define SQN_H

#include <cstdio>
#include <math.h>
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <iomanip>
#include "helpers.h"
#include "method_base.h"
#include "problem_base.h"

using namespace std;
// Stochastic limited-memory quasi-newton methods (Stochastic L-BFGS) algorithms.

class sqn: public method {  // need to reduce the memory use for different methods.
public:
    size_t l, p, batch_size, batch_size_hess, memory_size = 0, L, M;
    int sub_mode, step_alg, insert_at=-1;
    char method_name[15];
    double step_init, eps;
    Options *options;
    problem *prob;
//     double *gradient = NULL;
    
    double **s_array = NULL;
    double **y_array = NULL;
    double *w;
    
    // auxiliary variables helpful during the two loop recursion.
    double *q = NULL;
    double *a = NULL;
	double *rk = NULL;
    double *R = NULL;

public:
    sqn(problem *pr, Options *opts);
    ~sqn();
    void solve(double *w, outputs *output);

private:
    void lbfgs_two_loop_recursion(double *grad, double *HessGrad);
    void storeCurvature(double *w, double *w0, double *Hv);
    void storeCurvature(double *w, double *w0, double *grad_new, double *grad_old);
};

#endif