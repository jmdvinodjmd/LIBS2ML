/**
 * helpers.cpp
 * Usage: This file contains some auxiliary definitions, classes and functions.
 * This file is part of LIBS2ML.
 * Created by V. K. Chauhan on Sept. 15, 2018
 * Last Modified by V. K. Chauhan on Apr. 15, 2019
 **/

#ifndef HELPERS_H
#define HELPERS_H


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

// Add one entry for each method, problem and learning rate technique
enum {TRON, STRON, STRON_SVRG, STRON_PCG, STRON_PCG_PROG, SQN, SVRG_LBFGS, SVRG_SQN, LBFGS, NEWTON_CG};
enum {LOGISTIC_REGRESSION_L2R, SVM_L2R_L2L};
enum {BACKTRACKING_LINE_SEARCH_BATCH, BACKTRACKING_LINE_SEARCH_FULL, FIXED, FIXED_EPOCH};

struct Options {
    size_t l, p, l_test;
    size_t max_iters;
    size_t batch_size, cg_iters;
	int method_type, step_alg;
	double eps;
	double lambda;
};
struct feature_node {
	int index;
	double value;
};
struct outputs {
	double *costs;
    double *accuracy;
    double *time;
    int *grad_calc;
};
struct inputs {
	size_t l, l_test, p;
    double *w;
	double *y, *y_test;
	struct feature_node **x, **x_test;
	double bias;            /* < 0 if no bias term */
};


// This class used from LIBLINEAR to keep the data formats same.
 double nrm2_sq(const feature_node *x);

 double dot(const double *s, const feature_node *x);

 void axpy(const double a, const feature_node *x, double *y);

double dot_(size_t p, double *x, double *y);
double norm_(size_t p, double *y);
void axpy_(size_t p, const double a, double *x, double *y);
#endif