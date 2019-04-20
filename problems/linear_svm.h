/**
 *
 * Usage: This file defines a class for l2-regularized l2-loss SVM problem (binary)
 * containing methods, like gradient and Hessian etc., required to work with the problem.
 *
 *          SVM (Support Vector Machine):
 *
 *                  min F(w) = 1/n * sum_i^n f_i(w),           
 *          where 
 *                  f_i(w) = max(0, 1 - y_i * (w'*x_i)) + lambda/2 * w^2,
 *
 * where w \in \mathbb{R}^d is parameter vector of dimension d and lambda is regularization coefficinet.
 *
 * Created by V. K. Chauhan on Sept. 17, 2018
 * Modified by V. K. Chauhan on Apr. 15, 2019
 *
**/
#ifndef LINEAR_SVM_H
#define LINEAR_SVM_H

#include <math.h>
#include <random>
#include <time.h>
#include "helpers.h"
#include "problem_base.h"


// class definition for SVM problem
class svm_l2r_l2l: public problem {
public:
    inputs *input;
    Options *options;
    double *w;
    size_t l, l_test, batch_size=10, batch_size_hess=200;
    double lambda;
    
public:
    svm_l2r_l2l(inputs *input, Options *options);
    ~svm_l2r_l2l();
    
    double cost_full(double *w);
    double cost_batch(double *w, size_t start_index, size_t end_index);
    double cost_test(double *w);
	void gradient_full(double *w, double *gradient);
    void gradient_batch(double *w, size_t start_index, size_t end_index, double *gradient);
	void Hessian_vector_product(double *s, double *diag, double *Hess_vec);
    void Hessian_vector_product_batch(double *s, double *diag, double *Hess_vec, 
            size_t start_index, size_t end_index);
    void Hessian_vector_product_batch_sqn(double *w, double *u_new, double *u_old,
        double *Hess_vec, size_t start_index, size_t end_index);

	void get_diag_pre_condH(double *w, double *diag, double *pre_cond_H);
    void get_diag_pre_condH_batch(double *w, double *diag, double *pre_cond_H, size_t start_index, size_t end_index);
    double accuracy(double *w);
    
    void initilize_perm_idx();
    void randomize();
    void randomize(size_t);
    void hess_sampling(size_t *start_index_hess, size_t *end_index_hess);
    void get_diag(double *w, double *diag, size_t start_index, size_t end_index);
    void get_diag(double *w, double *diag);
    double backtracking_mb_ls(double *w, double *g0, double * p, double rho, double c,
        size_t start_index, size_t end_index);
    double backtracking_mb_ls(double *w, double *g0, double rho, double c,
        size_t start_index, size_t end_index);
    double backtracking_ls(double *w, double *g0, double *Hg, double rho, double c);
    double backtracking_ls(double *w, double *g, double rho, double c);
};
#endif