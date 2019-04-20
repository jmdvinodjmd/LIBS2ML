/**
 *
 * Usage: This file defines the stron_pcg_prog class for the STRON-PCG-PROG method (Chauhan et al 2018) in C++.
 *          It uses STRON with pcg subproblem solver and progressive subsampling.
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
#ifndef STRON_PCG_PROG_H
#define STRON_PCG_PROG_H

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

class stron_pcg_prog: public method {
public:
    size_t l, p, batch_size;
    Options *options;
    problem *prob;
    double *gradient = NULL;
    double *s = NULL;
    double *r = NULL;

    double *d = NULL;
	double *Hd = NULL;
    double *z = NULL;
    double *diag = NULL;
    double *pre_cond_H = NULL;
    double *w;
    
    // variables to be passed to the trcgbatch()
    double delta;
    bool at_boundary;
    size_t cg_iters;

public:
    stron_pcg_prog(problem *pr, Options *opts);
    ~stron_pcg_prog();
    void solve(double *w, outputs *output);

private:
    size_t trpcg_batch(size_t start_index, size_t end_index);
    size_t trpcg();
    double uTMv(double *u, double *M, double *v);
};
#endif