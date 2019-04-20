/**
 *
 * Usage: This file defines the newtoncg class for the Newton-CG method (Byrd et al 2011).
 * This is subsampled Newton CG method with progressive sampling addition to the main method.
﻿@article{Byrd2011,
	author = {Byrd, R. and Chin, G. and Neveitt, W. and Nocedal, J.},
	title = {On the Use of Stochastic Hessian Information in Optimization Methods for Machine Learning},
	journal = {SIAM Journal on Optimization},
	volume = {21},
	number = {3},
	pages = {977-995},
	year = {2011},
	doi = {10.1137/10079923X}
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

#ifndef NEWTON_CG_H
#define NEWTON_CG_H

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

class newtoncg: public method {
public:
    size_t l, p, batch_size;
    int step_alg;
    double step_init, eps;
    Options *options;
    problem *prob;
    double *gradient = NULL;
    double *s = NULL;
    double *r = NULL;

    double *d = NULL;
	double *Hd = NULL;
    double *diag = NULL;
    double *w;
    
    // variables to be passed to the trcgbatch()
    size_t cg_iters;

public:
    newtoncg(problem *pr, Options *opts);
    ~newtoncg();
    void solve(double *w, outputs *output);

private:
    size_t cg_batch(size_t start_index, size_t end_index);
    double uTMv(double *u, double *M, double *v);
};

#endif