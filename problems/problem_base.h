/**
 * problem_base.cpp
 * Usage: This file defines base `problem' class for different problems. Each new problem class should extend this class.
 *
 * This file is part of LIBS2ML.
 * Created by V. K. Chauhan on Sept. 15, 2018
 * Last Modified by V. K. Chauhan on Apr. 15, 2019
 **/

#ifndef PROBLEM_BASE_H
#define PROBLEM_BASE_H

#include "helpers.h"
#include <stdio.h>

class problem {
public:
    size_t p;
    feature_node **x;
    double *y;
    feature_node **x_test;
    double *y_test;
    size_t *perm_index = NULL;
    
    double *w_temp = NULL;
    double *u_temp = NULL;
    
public:
    virtual ~problem() {}
    virtual double accuracy(double *w) = 0;
    
	virtual double cost_full(double *w) = 0;
    virtual double cost_batch(double *w, size_t start_index, size_t end_index) = 0;
    virtual double cost_test(double *w) = 0;
	virtual void gradient_full(double *w, double *gradient) = 0;
    virtual void gradient_batch(double *w, size_t start_index, 
            size_t end_index, double *gradient) = 0;
	virtual void Hessian_vector_product(double *s, double *diag, double *Hess_vec) = 0;
    virtual void Hessian_vector_product_batch(double *s, double *diag, double *Hess_vec, 
            size_t start_index, size_t end_index) = 0;
    virtual double backtracking_mb_ls(double *w, double *g0, double * p, double rho, double c,
        size_t start_index, size_t end_index) = 0;
    virtual double backtracking_mb_ls(double *w, double *g, double rho, double c,
        size_t start_index, size_t end_index) = 0;
    virtual double backtracking_ls(double *w, double *g0, double *Hg, double rho, double c) = 0;
    virtual double backtracking_ls(double *w, double *g0, double rho, double c) = 0;
    virtual void initilize_perm_idx() = 0;
    virtual void randomize() = 0;
    virtual void randomize(size_t) = 0;
    virtual void get_diag(double *w, double *diag, size_t start_index, size_t end_index) = 0;
    virtual void get_diag(double *w, double *diag) = 0;
    virtual void Hessian_vector_product_batch_sqn(double *w, double *u_new, double *u_old,
        double *Hess_vec, size_t start_index, size_t end_index) = 0;
	virtual void get_diag_pre_condH(double *w, double *diag, double *pre_cond_H) = 0;
    virtual void get_diag_pre_condH_batch(double *w, double *diag, double *pre_cond_H, size_t start_index, size_t end_index) = 0;
    virtual void hess_sampling(size_t *start_index_hess, size_t *end_index_hess) = 0;
};

#endif