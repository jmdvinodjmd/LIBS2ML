/**
 *
 * Usage: This file defines the stron class for the STRON method (Chauhan et al 2018).
 * This method need tuning of rate of increasing the mini-batch size, in addition to other params.
 *
@article{Chauhan2018STRON,
	title = {{Stochastic Trust Region Inexact Newton Method for Large-scale Machine Learning}},
	journal = {arXiv},
	arxivId = {1812.10426},
	author = {Chauhan, Vinod Kumar and Sharma, Anuj and Dahiya, Kalpana},
	eprint = {1812.10426},
	month = {dec},
	url = {http://arxiv.org/abs/1812.10426},
	year = {2018}
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
#ifndef STRON_H
#define STRON_H

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

class stron: public method {
public:
    size_t l, p, batch_size;
    Options *options;
    problem *prob;
    double eps;
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
    size_t cg_iters;

public:
    stron(problem *pr, Options *opts);
    ~stron();
    void solve(double *w, outputs *output);

private:
    size_t trcg_batch(size_t start_index, size_t end_index);
    size_t trcg();
    double uTMv(double *u, double *M, double *v);
};
#endif