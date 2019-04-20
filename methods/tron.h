/**
 *
 * Usage: This file defines the tron class for the TRON method in C++ (Hsia et al. 2018).
﻿@InProceedings{Hsia2018,
	title = 	 {Preconditioned Conjugate Gradient Methods in Truncated Newton Frameworks for Large-scale Linear Classification},
	author = 	 {Chih-Yang Hsia and Wei-Lin Chiang and Chih-Jen Lin},
	booktitle = 	 {ACML},
	year = 	 {2018},
	publisher = 	 {PMLR}
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
#ifndef TRON_H
#define TRON_H

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

class tron: public method {
public:
    size_t l, p, cg_iters;
    Options *options;
    problem *prob;
    double eps;
    
    double *d = NULL;
	double *Hd = NULL;
	double *z = NULL;
    
    double *gradient = NULL;
    double *s = NULL;
    double *r = NULL;
    
public:
    tron(problem *pr, Options *opts);
    ~tron();
    void solve(double *w, outputs *output);

    
private:
    size_t trpcg(double *pre_cond_H, double *diag, 
            double delta, size_t cg_iters, bool *at_boundary);
    double uTMv(double *u, double *M, double *v);
};

#endif