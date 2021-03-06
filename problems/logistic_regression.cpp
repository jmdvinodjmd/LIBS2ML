/**
 *
 * Usage: This file defines a class for l2-regularized logistic regression problem (binary)
 * containing methods, like gradient and Hessian etc., required to work with the problem.
 *
 *          Logistic Regression problem:
 *
 *                  min F(w) = 1/n * sum_i^n f_i(w),           
 *          where 
 *                  f_i(w) = log(1 + exp(-y_i * (w'*x_i))) + lambda/2 * w^2,
 *
 * where w \in \mathbb{R}^d is parameter vector of dimension d and lambda is regularization coefficinet.
 *
 * Created by V. K. Chauhan on Sept. 17, 2018
 * Modified by V. K. Chauhan on Apr. 15, 2019
 *
**/

#include <math.h>
#include <random>
#include <time.h>
#include "helpers.h"
#include "problem_base.h"
#include "logistic_regression.h"

// class definition for logistic regression problem

logistic_regression_l2r::logistic_regression_l2r(inputs *in, Options *opts) {
    input = in;
    options = opts;
    l = options->l;
    l_test = options->l_test;
    p = options->p;
    lambda = options->lambda;
    x = input->x;
    y = input->y;
    x_test = input->x_test;
    y_test = input->y_test;

    w_temp = new double[p]; // these can be optimized by limiting to SQN.
    u_temp = new double[p];
//     batch_size = opts->batch_size;
//     batch_size_hess = opts->batch_size_hess;
    perm_index = new size_t[l];   // can be optimized, as it is required in some cases.
}

logistic_regression_l2r::~logistic_regression_l2r() {
    if(perm_index!=NULL) {
        delete[] perm_index;
    }
    if(w_temp!=NULL) {
        delete[] w_temp;
    }
    if(u_temp!=NULL) {
        delete[] u_temp;
    }
}

// This is used to calculate objective function over the whole train dataset.
double logistic_regression_l2r::cost_full(double *w) {
    double cost = 0, w_dot = 0, temp;
    for(size_t i=0; i<l; ++i) {
        temp = dot(w, x[i]); 
        temp = y[i]*temp;
        if(temp>=0)
            cost += log(1 + exp(-temp));
        else    //??
            cost += -temp + log(1 + exp(temp));
    }
    cost = cost/(double) l;
    for(size_t k=0; k<p; k++) {
        w_dot += w[k]*w[k];
    }
    cost += lambda * w_dot/2.0;
    return cost;
}

// This is used to calculate the objective function over mini-batch of train data points.
double logistic_regression_l2r::cost_batch(double *w, size_t start_index, 
        size_t end_index) {
    double cost = 0, w_dot = 0, temp;
    size_t batch_size = end_index - start_index + 1;
    for(size_t i=start_index; i<=end_index; ++i) {
        temp = dot(w, x[perm_index[i]]);
        temp = y[perm_index[i]]*temp;
        if(temp>=0) {
            cost += log(1 + exp(-temp));
        } else {    // why this collabery??
            cost += -temp + log(1 + exp(temp));
        }
    }
    cost = cost/(double) batch_size + lambda * dot_(p, w, w)/2.0;
    return cost;
}

// This is used to calculate objective function over the test dataset.
double logistic_regression_l2r::cost_test(double *w) {
    double cost = 0, w_dot = 0, temp;
    for(size_t i=0; i<l_test; ++i) {
        temp = dot(w, x_test[i]); 
        temp = y_test[i]*temp;
        if(temp>=0)
            cost += log(1 + exp(-temp));
        else    // why this collabery??
            cost += -temp + log(1 + exp(temp));
    }
    cost = cost/(double) l_test;
    for(size_t k=0; k<p; k++) {
        w_dot += w[k]*w[k];
    }
    cost += lambda * w_dot/2.0;
    return cost;
}

// This is used to calculate gradient over the whole dataset.
void logistic_regression_l2r::gradient_full(double *w, double *gradient) {
    double temp, yi;
    size_t i;
    for(i=0; i<p; i++) { gradient[i] = 0; }
    for(i=0; i<l; i++) {
        yi = y[i];
        temp = dot(w, x[i]);
        temp = exp(-yi*temp);
        temp = -yi*temp/(1+temp);
        axpy(temp, x[i], gradient);        
    }
    for(i=0; i<p; i++) {
        gradient[i] = gradient[i]/(double)l + lambda*w[i];
    }
}

// This is used to calculate gradient function over the mini-batch.
void logistic_regression_l2r::gradient_batch(double *w, size_t start_index, 
        size_t end_index, double *gradient) {
    double temp, yi;
    size_t i, batch_size;
    batch_size = end_index-start_index+1;
    for(i=0; i<p; i++) { gradient[i] = 0; }
    for(i=start_index; i<=end_index; i++) {
        yi = y[perm_index[i]];
        temp = dot(w, x[perm_index[i]]);
        temp = exp(-yi*temp);
        temp = -yi*temp/(1+temp);
        axpy(temp, x[perm_index[i]], gradient);        
    }
    for(i=0; i<p; i++) {
        gradient[i] = gradient[i]/(double)batch_size + lambda*w[i];
    }
}

// This is used to calculate Hessian-vector product over the whole dataset
void logistic_regression_l2r::Hessian_vector_product(double *s, double *diag, double *Hess_vec) {
    double temp1;
    size_t i;
	for(i=0; i<p; i++) { Hess_vec[i] = 0; }
	for(i=0; i<l; i++) {
        // diagonal getting changed every time
        // it should not change until the parameter vector is changed.
        temp1 = dot(s, x[i]);
//         temp2 = 1.0/(1 + exp(-y[i]*temp1));
// 		diag = temp2*(1-temp2);
		temp1 = diag[i]*temp1;
		axpy(temp1, x[i], Hess_vec);
	}
	for(i=0; i<p; i++) {
		Hess_vec[i] = Hess_vec[i]/(double)l + lambda*s[i];
    }
}

// This is used to calculate Hessian-vector product over the mini-batch.
void logistic_regression_l2r::Hessian_vector_product_batch(double *s, double *diag,
        double *Hess_vec, size_t start_index, size_t end_index) {
    double temp;
    size_t batch_size;
    batch_size = end_index-start_index+1;
    for(size_t i=0; i<p; i++) { Hess_vec[i] = 0; }
    for(size_t i=start_index, j=0; i<=end_index; i++, j++) {
        temp = dot(s, x[perm_index[i]]);
		temp = diag[j]*temp;
		axpy(temp, x[perm_index[i]], Hess_vec);
	}
	for(size_t i=0; i<p; i++) {
		Hess_vec[i] = Hess_vec[i]/(double)batch_size + lambda*s[i];
    }
}
// This is used to calculate Hessian-vector product over the mini-batch.
void logistic_regression_l2r::Hessian_vector_product_batch_sqn(double *w, double *u_new, double *u_old,
        double *Hess_vec, size_t start_index, size_t end_index) {
    double temp, diag;
    size_t batch_size;
    batch_size = end_index-start_index+1;
    
    for(size_t i=0; i<p; i++) { Hess_vec[i] = 0; }
    for(size_t i=start_index; i<=end_index; i++) {
        diag = dot(w, x[perm_index[i]]);
        diag = 1.0/(1 + exp(-y[perm_index[i]]*diag));
        diag = diag*(1-diag);
        
        for(size_t j=0; j<p; ++j) {
            u_temp[j] = u_new[j] - u_old[j];
        }
        temp = dot(u_temp, x[perm_index[i]]);
		temp = diag*temp;
		axpy(temp, x[perm_index[i]], Hess_vec);
	}
	for(size_t i=0; i<p; i++) {
		Hess_vec[i] = Hess_vec[i]/(double)batch_size + lambda*(u_new[i]-u_old[i]);
    }
}

// This is used to calculate diagonal preconditioner over the whole dataset.
// to be used by Inexact Newton methods
void logistic_regression_l2r::get_diag_pre_condH(double *w, double *diag, double *pre_cond_H) {
    size_t i;
    double temp1, temp2;
    // first calculating the diagonal matrix
    for(i=0; i<l; i++) {
        temp1 = dot(w, x[i]);
        temp2 = 1.0/(1 + exp(-y[i]*temp1));
		diag[i] = temp2*(1-temp2);
	}
	for(i=0; i<p; i++) {
		pre_cond_H[i] = lambda;
    }
	for(i=0; i<l; i++) {
		feature_node *fn = x[i];
		while (fn->index!=-1) {
			pre_cond_H[fn->index-1] += fn->value*fn->value*diag[i];
			fn++;
		}
	}
}

// This is used to calculate diagonal preconditioner over the mini-batch.
// to be used by Inexact Newton methods
void logistic_regression_l2r::get_diag_pre_condH_batch(double *w, double *diag, 
        double *pre_cond_H, size_t start_index, size_t end_index) {
    size_t i, j;
    double temp1, temp2;
    size_t batch_size = end_index - start_index + 1;
    // first calculating the diagonal matrix
    for(i=start_index, j=0; i<=end_index; i++, j++) {
        temp1 = dot(w, x[perm_index[i]]);
        temp2 = 1.0/(1 + exp(-y[perm_index[i]]*temp1));
		diag[j] = temp2*(1-temp2);
	}
	for(i=0; i<p; i++) {
		pre_cond_H[i] = lambda;
    }
	for(i=start_index, j=0; i<=end_index; i++, j++) {
		feature_node *fn = x[perm_index[i]];
		while (fn->index!=-1) {
			pre_cond_H[fn->index-1] += fn->value*fn->value*diag[j];
			fn++;
		}
	}
}

// This is used to calculate accuracy of the model.
double logistic_regression_l2r::accuracy(double *w) {
    double correct = 0, temp;
    for(size_t i=0; i<l_test; ++i) {
        temp = dot(w, x_test[i]);
        temp = 1.0/(1 + exp(-temp));
        if(((temp > 0.5) && (y_test[i] > 0)) || (((temp <= 0.5) && (y_test[i] <= 0)))) {
            correct = correct + 1;
        }
    }
    return correct/(double)l_test;
}

void logistic_regression_l2r::initilize_perm_idx() {    
    for(size_t i=0; i<l; ++i) { 
        perm_index[i] = i;
    }
}

void logistic_regression_l2r::randomize() {
//     clock_t tic, toc;
 // Randomizing the data points.
//      std::cout<<"randomization started...\n"; 
//     tic = clock();
    size_t j, temp;
    for(size_t i=0; i < l; i++) {
        j = i + (size_t)(rand()% (l-i));
        temp = perm_index[i];
        perm_index[i] = perm_index[j];
        perm_index[j] = temp;
    }
//     toc = clock();
//     std::cout<<"Time- "<<(toc - tic)/(double) CLOCKS_PER_SEC<<"\n"; 
    
//     tic = clock();
//     // alternate method: very slow
//     std::random_shuffle(&perm_index[0], &perm_index[l-1]);
//     toc = clock();
//     std::cout<<"Time- "<<(toc - tic)/(double) CLOCKS_PER_SEC<<"\n";
}

void logistic_regression_l2r::randomize(size_t size) {
//     clock_t tic, toc;
 // Randomizing the data points.
    size_t j, temp;
    for(size_t i=0; i < size; i++) {
        j = i + (size_t)(rand()% (l-i));
        temp = perm_index[i];
        perm_index[i] = perm_index[j];
        perm_index[j] = temp;
    }
}

// This is used for sampling data points for Hessian.
void logistic_regression_l2r::hess_sampling(size_t *start_index_hess, size_t *end_index_hess) {
    size_t m;
    m =(size_t) ceil((double)l/batch_size_hess);    
    *start_index_hess = batch_size_hess*(rand()%(m-1));
    *end_index_hess = *start_index_hess + batch_size_hess - 1;
}

// This is used for calculatig thd diagonal matrix over the mini-batch
void logistic_regression_l2r::get_diag(double *w, double *diag, size_t start_index, size_t end_index) {
    size_t  j=0;
    double temp;
    for(size_t i=start_index; i<=end_index; i++) {
        temp = dot(w, x[perm_index[i]]);
        temp = 1.0/(1 + exp(-y[perm_index[i]]*temp));
//         std::cout<<temp;
		diag[j++] = temp*(1-temp);
	}
}

// This is used for calculatig thd diagonal matrix over the whole dataset.
void logistic_regression_l2r::get_diag(double *w, double *diag) {
    double temp;
    for(size_t i=0; i<l; i++) {
        temp = dot(w, x[i]);
        temp = 1.0/(1 + exp(-y[i]*temp));
//         std::cout<<temp;
		diag[i] = temp*(1-temp);
	}
}

// This is used for performing stochastic backtracking line search over the mini-batch.
double logistic_regression_l2r::backtracking_mb_ls(double *w, double *g0, double *Hg, double rho, double c,
        size_t start_index, size_t end_index) {
    // Stochastich Backtracking line search
    double alpha = 1, f0, fk, g0Tp;
    size_t counter = 0;
    for(size_t i=0; i<p; ++i) { w_temp[i] = w[i]; }
    f0 = cost_batch(w_temp, start_index, end_index);
    for(size_t i=0; i<p; ++i) { w_temp[i] = w_temp[i] + alpha * Hg[i]; }
    fk = cost_batch(w_temp, start_index, end_index);
    g0Tp = c * dot_(p, g0, Hg);
    
    // repeat until the Armijo condition meets
    while(fk > f0 + alpha * g0Tp) {
      alpha = rho * alpha;
      for(size_t i=0; i<p; ++i) { 
          w_temp[i] = w[i] + alpha * Hg[i]; 
      }
      fk = cost_batch(w_temp, start_index, end_index);
      counter = counter + 1;
      // added to terminate the loop
      if(counter > 10) {
          break;
      }
    }
    return alpha;
}

// This is used for performing stochastic backtracking line search over the mini-batch.
double logistic_regression_l2r::backtracking_mb_ls(double *w, double *g, double rho, double c,
        size_t start_index, size_t end_index) {
    // Stochastich Backtracking line search
    double alpha = 1, f0, fk, g0Tp;
    size_t counter = 0;
    for(size_t i=0; i<p; ++i) { w_temp[i] = w[i]; }
    f0 = cost_batch(w_temp, start_index, end_index);
    for(size_t i=0; i<p; ++i) { w_temp[i] = w_temp[i] - alpha * g[i]; }
    fk = cost_batch(w_temp, start_index, end_index);
    g0Tp = c * dot_(p, g, g);
    
    // repeat until the Armijo condition meets
    while(fk > f0 + alpha * g0Tp) {
      alpha = rho * alpha;
      for(size_t i=0; i<p; ++i) {
          w_temp[i] = w[i] - alpha * g[i];
      }
      fk = cost_batch(w_temp, start_index, end_index);
      counter = counter + 1;
      // added to terminate the loop
      if(counter > 10) {
          break;
      }
    }
    return alpha;
}

// This is used for performing backtracking line search over the whole dataset.
double logistic_regression_l2r::backtracking_ls(double *w, double *g0, double *Hg, double rho, double c) {    
    // Backtracking line search
    double alpha = 1, f0, fk, g0Tp;
    size_t counter = 0;
//     for(size_t i=0; i<p; ++i) { w_temp[i] = w[i]; }
    f0 = cost_full(w);
    for(size_t i=0; i<p; ++i) { w_temp[i] = w[i] + alpha * Hg[i]; }
    fk = cost_full(w_temp);
    g0Tp = c * dot_(p, g0, Hg);
//     std::cout<<"f0: "<<f0<<" fk: "<<fk<<" goTp "<<g0Tp<<" c "<<c<<" f0 + alpha * g0Tp: "<<f0 + alpha * g0Tp<<"\n";
    // repeat until the Armijo condition meets
    while(fk > f0 + alpha * g0Tp) {
      alpha = rho * alpha;
      for(size_t i=0; i<p; ++i) { 
          w_temp[i] = w[i] + alpha * Hg[i]; 
      }
      fk = cost_full(w_temp);
      counter = counter + 1;
      // added to terminate the loop
      if(counter > 10) {
          break;
      }
    }
//     std::cout<<"alpha: "<<alpha<<"\n";
    return alpha;
}

// This is used for performing backtracking line search over the whole dataset.
double logistic_regression_l2r::backtracking_ls(double *w, double *g, double rho, double c) {
    // Stochastich Backtracking line search
    double alpha = 1, f0, fk, g0Tp;
    size_t counter = 0;
//     for(size_t i=0; i<p; ++i) { w_temp[i] = w[i]; }
    f0 = cost_full(w);
    for(size_t i=0; i<p; ++i) { w_temp[i] = w[i] - alpha * g[i]; }
    fk = cost_full(w_temp);
    g0Tp = c * dot_(p, g, g);
    
    // repeat until the Armijo condition meets
    while(fk > f0 + alpha * g0Tp) {
      alpha = rho * alpha;
      for(size_t i=0; i<p; ++i) {
          w_temp[i] = w[i] - alpha * g[i];
      }
      fk = cost_full(w_temp);
      counter = counter + 1;
      // added to terminate the loop
      if(counter > 20) {
          break;
      }
    }
    return alpha;
}