/**
 *
 * Usage: This file defines the stron_svrg class for the STRON-SVRG in C++.
 *        It uses variance reduction using SVRG and progressive batching for Hessian calculations.
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
 * Modified by V. K. Chauhan on Apr. 17, 2019
 *
**/

#ifndef STRON_SVRG_H
#define STRON_SVRG_H

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

class stron_svrg: public method {
public:
    size_t l, p, batch_size, batch_size_hessian;
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
    double delta;
    bool at_boundary;
    size_t cg_iters = 10;

public:
    stron_svrg(problem *pr, Options *opts);
    ~stron_svrg();
    void solve(double *w, outputs *output);

private:
    size_t trcg_batch();
    double uTMv(double *u, double *M, double *v);
};

#endif