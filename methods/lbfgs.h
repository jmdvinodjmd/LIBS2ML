/**
 *
 * Usage: This defines lbfgs class for LBFGS method in C++.
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

#ifndef LBFGS_H
#define LBFGS_H

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

class lbfgs: public method {
public:
    size_t l, p, memory_size=0, L;
    int step_alg, insert_at=-1;
    double step_init;
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
    lbfgs(problem *pr, Options *opts);
    ~lbfgs();
    void solve(double *w, outputs *output);

private:
    void lbfgs_two_loop_recursion(double *grad, double *HessGrad);
    void storeCurvature(double *w, double *w0, double *Hv);
    void storeCurvature(double *w, double *w0, double *grad_new, double *grad_old);
};

#endif