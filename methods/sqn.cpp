/**
 *
 * Usage: This file defines the sqn class for the SQN, SVRG_LBFGS and SVRG_SQN in C++.
﻿﻿@article{SQN,
	author    = {Richard H. Byrd and S. L. Hansen and Jorge Nocedal and	Yoram Singer},
	title     = {A Stochastic Quasi-Newton Method for Large-Scale Optimization},
	journal   = {{SIAM} Journal on Optimization},
	volume    = {26},
	number    = {2},
	pages     = {1008-1031},
	year      = {2016}
}
@inproceedings{SVRG_SQN,
	title={A Linearly-Convergent Stochastic L-BFGS Algorithm},
	author={Philipp Moritz and Robert Nishihara and Michael I. Jordan},
	booktitle={AISTATS},
	year={2016}
}
﻿@inproceedings{SVRG_LBFGS,
	title={Accelerating SVRG via second-order information},
	author={Kolte, Ritesh and Erdogdu, Murat and Ozgur, Ayfer},
	booktitle={NIPS Workshop on Optimization for Machine Learning},
	year={2015}
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

#include "sqn.h"

sqn::sqn(problem *pr, Options *opts) {
    options = opts;
    prob = pr;
    l = opts->l;
    p = opts->p;
    batch_size = opts->batch_size;
    batch_size_hess = batch_size;
    eps = options->eps;
    
    L = 5;
    M = 5;
    sub_mode =  opts->method_type;  // SVRG_SQN  SQN     SVRG_LBFGS
    step_alg = BACKTRACKING_LINE_SEARCH_BATCH; //opts->step_alg;// FIXED_EPOCH FIXED BACKTRACKING_LINE_SEARCH_BATCH, BACKTRACKING_LINE_SEARCH_FULL
    step_init = 0.001;
  
    s_array = new double*[M];
    for(size_t i=0; i<M; ++i) {
        s_array[i] = new double[p];
    }
	y_array = new double*[M];
    for(size_t i=0; i<M; ++i) {
        y_array[i] = new double[p];
    }
    
    q = new double[p];
    a = new double[M];
    rk = new double[M];
    R = new double[p];
    
    switch(opts->method_type) {
        case SVRG_SQN:
            strcpy(method_name, "SVRG_SQN");
            break;
        case SVRG_LBFGS:
            strcpy(method_name, "SVRG_LBFGS");
            break;
        case SQN:
            strcpy(method_name, "SQN");
            break;
    }
}
sqn::~sqn() {
    
    if(s_array!=NULL) {
        for(size_t i=0; i<M; ++i) {
            delete[] s_array[i];
        }
        delete[] s_array;
    }
    if(y_array!=NULL) {
        for(size_t i=0; i<M; ++i) {
            delete[] y_array[i];
        }
        delete[] y_array;
    }
    if(q!=NULL)
        delete[] q;
    if(a!=NULL)
        delete[] a;
    if(rk!=NULL)
        delete[] rk;
    if(R!=NULL)
        delete[] R;
}

void sqn::solve(double *wts, outputs *output) {
    double *costs, *time, *accuracy;
    size_t max_iters, iter, total_iter, number_of_batches;
    size_t start_index, end_index, start_index_hess, end_index_hess;
    bool run_flag = true, info=false, test_cost_flag = false;
    double step, f_val, gnorm, gnorm0;

    clock_t tic, toc, t1, t2; // variables to record the running time.
    double *grad = new double[p];
    double *full_grad = new double[p];
    double *full_grad_new = new double[p];
    double *grad_w0 = new double[p];
    double *Hg = new double[p];
    double *Hv = NULL;
    
    double *u_new=NULL;
    double *u_old=NULL;
    
    if(sub_mode!=SVRG_LBFGS) {
        Hv = new double[p];
        u_new = new double[p];
        u_old = new double[p];
    }
    
    double *w0 = new double[p];

    w = wts;    // initialize the w of ston class to use in other functions.
    max_iters = options->max_iters;
    costs = output->costs;
    time = output->time;
    accuracy = output->accuracy;
    
    // initialise the output
    iter = 0;
    total_iter = 0;
    time[iter] = 0;
    if(test_cost_flag) {
        f_val = prob->cost_test(w);
    } else {
        f_val = prob->cost_full(w);
    }
    costs[iter] = f_val;
    accuracy[iter] = prob->accuracy(w);
    prob->initilize_perm_idx();
    prob->gradient_full(w, grad);
    gnorm0 = norm_(p, grad);
    
    srand(2018);
    if(sub_mode!=SVRG_LBFGS) {
        for(size_t ii=0; ii<p; ++ii) { u_old[ii] = w[ii]; }
        for(size_t ii=0; ii<p; ++ii) { u_new[ii] = 0; }
    }
    
    std::cout.setf(ios::fixed);
    std::cout<<method_name<<" iter:"<<setw(3)<<iter<<" f: "<<setprecision(15)<< costs[iter]<<", accuracy: "<<setprecision(4)<<accuracy[iter]<<", Time: "<<setprecision(5)<< time[iter] <<std::endl;
    
    // total gradient calculations during one epoch(grads/l):
    switch(sub_mode) {
        case SVRG_SQN:
        case SVRG_LBFGS:
            output->grad_calc[0] = 3;
            break;
        case SQN:
            output->grad_calc[0] = 1;
            break;
    }
    
    number_of_batches = (size_t) ceil((double)l/(batch_size));

    // set start time
    tic = clock();
    
    while(iter < max_iters) { // outer iterations        
        // Randomizing the data points.
        prob->randomize();        
        if((sub_mode==SVRG_SQN) || (sub_mode==SVRG_LBFGS)) {
            prob->gradient_full(w, full_grad_new);
        }
        if(sub_mode==SVRG_LBFGS) {
            if (iter > 0) {
                // store curvature pair
                storeCurvature(w, w0, full_grad_new, full_grad);
            }
        }            
        if((sub_mode==SVRG_SQN) || (sub_mode==SVRG_LBFGS)) {
            // store w for SVRG
            for(size_t i=0; i<p; ++i) { w0[i] = w[i]; }
            for(size_t i=0; i<p; ++i) { full_grad[i] = full_grad_new[i]; }
        }
        
        for(size_t iter_batch=1; iter_batch<=number_of_batches; ++iter_batch) {
            start_index = (iter_batch-1) * batch_size;
            end_index = start_index + batch_size - 1;
            if(end_index >= l) {
                end_index = l-1;
            }
            
            // calculate gradient
            prob->gradient_batch(w, start_index, end_index, grad);
            if((sub_mode==SVRG_SQN) || (sub_mode==SVRG_LBFGS)) {
                prob->gradient_batch(w0, start_index, end_index, grad_w0);
                for(size_t i=0; i<p; ++i) {
                    grad[i] = grad[i] - grad_w0[i] + full_grad[i];
                }
            }
            if(iter > 0) {
                // perform LBFGS two loop recursion
                lbfgs_two_loop_recursion(grad, Hg);
                if(step_alg==BACKTRACKING_LINE_SEARCH_BATCH) {
                    prob->hess_sampling(&start_index_hess, &end_index_hess);
                    double rho = 0.5;
                    double c = 1e-4;   // might need to adjust the value of c and change
                    // stochastic backtracking line search method
//                     step = prob->backtracking_ls(w, grad, Hg, rho, c);
                    step = prob->backtracking_mb_ls(w, grad, Hg, rho, c, start_index_hess, end_index_hess);
                } else if(step_alg==FIXED) {
                    step = step_init;
                } else {
                    step = step_init/(double)(iter+1);
                }
                axpy_(p, step, Hg, w);
            } else {
                if(step_alg==BACKTRACKING_LINE_SEARCH_BATCH) {
                    prob->hess_sampling(&start_index_hess, &end_index_hess);
                    // using the backtracking line search
                    double rho = 0.5;
                    double c = 1e-4;   // might need to adjust the value of c and change
                    // the backtracking algorithm.
                    // stochastic backtracking line search method
//                     step = prob->backtracking_ls(w, grad, rho, c);
                    step = prob->backtracking_mb_ls(w, grad, rho, c, start_index_hess, end_index_hess);
                } else if(step_alg==FIXED) {
                    step = step_init;
                } else if(FIXED_EPOCH) {
                    step = step_init/(double)(iter+1);
                }
                axpy_(p, -step, grad, w);
            }
            
            // calculate averaged w
            if((sub_mode==SVRG_SQN) || (sub_mode==SQN)) {
                axpy_(p, 1.0/L, w, u_new);
            }

            // update LBFGS vectors Hessian at every L iteration for 'SQN' or 'SVRG_SQN'
            // 'SVRG_LBFGS' does nothing because of L = Inf
            if((sub_mode!=SVRG_LBFGS) && (total_iter%L==0) && total_iter) {
                // calcluate Hessian-vector product using subsamples
                prob->hess_sampling(&start_index_hess, &end_index_hess);
                prob->Hessian_vector_product_batch_sqn(w, u_new, u_old, Hv, start_index_hess, end_index_hess);

                // store cavature pair
                // 'y' curvature pair is calculated from a Hessian-vector product.
                storeCurvature(u_new, u_old, Hv);                
                for(size_t ii=0; ii<p; ++ii) {
                    u_old[ii] = u_new[ii];
                    u_new[ii] = 0; 
                }
            }
            total_iter = total_iter + 1;        
        }
        iter = iter + 1;
        // measure elapsed time
        t1 = clock();
        if(test_cost_flag) {
            f_val = prob->cost_test(w);
        } else {
            f_val = prob->cost_full(w);
        }
        costs[iter] = f_val;
        accuracy[iter] = prob->accuracy(w);
        t2 = clock();
        toc = clock();
        time[iter] = (toc - tic - iter*(t2-t1))/(double) CLOCKS_PER_SEC;
        
        gnorm = norm_(p, grad);
        // display the output after the iteration
        std::cout<<method_name<<" iter:"<<setw(3)<<iter<<" f: "<<setprecision(15)<< costs[iter]<<" |g|: "<<setprecision(10)<< gnorm<<" Time: "<<setprecision(2)<< time[iter]<<" accuracy: "<<setprecision(4)<<accuracy[iter] <<std::endl;
        
        if (gnorm <= eps*gnorm0) {
            std::cout<<("reached required tolerance\n");
            run_flag = false;
            break;
        }
    }
            
    // clear memory
    if(grad!=NULL)
        delete[] grad;
    if(full_grad!=NULL)
        delete[] full_grad;
    if(full_grad_new!=NULL)
        delete[] full_grad_new;
    if(grad_w0!=NULL)
        delete[] grad_w0;
    if(Hg!=NULL)
        delete[] Hg;
    if(Hv!=NULL)
        delete[] Hv;        
    if(w0!=NULL)
        delete[] w0;
    if(u_new!=NULL)
        delete[] u_new;
    if(u_old!=NULL)
        delete[] u_old;
    std::cout<<method_name<<"/c++ completed...\n\n";
}

void sqn::lbfgs_two_loop_recursion(double *grad, double *HessGrad) {
// Two loop recursion algorithm for L-BFGS.
//
// Reference:
//       Jorge Nocedal and Stephen Wright,
//       "Numerical optimization,"
//       Springer Science & Business Media, 2006.
    
    double beta, Hk0;
    if(insert_at==-1) {
        for(size_t i=0; i<p; ++i) { HessGrad[i] = -grad[i]; }
    } else {
        for(size_t i=0; i<p; ++i) { q[i] = grad[i]; }
        for(int i=memory_size-1; i>=0; i--) {
            rk[i] = 1/dot_(p, y_array[i], s_array[i]);
            a[i] = rk[i]*dot_(p, s_array[i], q);
            axpy_(p, -a[i], y_array[i], q);
        }
        Hk0 = dot_(p, s_array[memory_size-1], y_array[memory_size-1])/dot_(p, y_array[memory_size-1], y_array[memory_size-1]);
        for(size_t i=0; i<p; ++i) {
            R[i] = Hk0*q[i];
        }        
        for(size_t jj=0; jj<memory_size; jj++) {
            beta = rk[jj]*dot_(p, y_array[jj], R);
            for(size_t i=0; i<p; ++i) {
                R[i] = R[i] + s_array[jj][i]*(a[jj] - beta);
            }
        }
        for(size_t i=0; i<p; i++) {
            HessGrad[i] = -R[i]; 
        }
    }
}

void sqn::storeCurvature(double *w, double *w0, double *grad_new, double *grad_old) {
    if(memory_size!=M) {
        memory_size++;
    }
    insert_at = (++insert_at) % M;
    for(size_t i=0; i<p; ++i) {
        s_array[insert_at][i] = w[i]-w0[i];
        y_array[insert_at][i] = grad_new[i]-grad_old[i];
    }
}

void sqn::storeCurvature(double *w, double *w0, double *Hv) {
    if(memory_size!=M) {
        memory_size++;
    }
    insert_at = (++insert_at) % M;
    for(size_t i=0; i<p; ++i) {
        s_array[insert_at][i] = w[i]-w0[i];
        y_array[insert_at][i] = Hv[i];
    }
}