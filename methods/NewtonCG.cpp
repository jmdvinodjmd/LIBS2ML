/**
 *
 * Usage: This file defines the newtoncg class for the Newton-CG method (Byrd et al 2011).
 * This is subsampled Newton CG method with progressive sampling addition to the main method.
﻿@article{Byrd2011,
	author = {Byrd, R. and Chin, G. and Neveitt, W. and Nocedal, J.},
	title = {On the Use of Stochastic Hessian Information in Optimization Methods for Machine Learning},
	journal = {SIAM Journal on Optimization},
	volume = {21},
	number = {3},
	pages = {977-995},
	year = {2011},
	doi = {10.1137/10079923X}
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

#include "NewtonCG.h"

newtoncg::newtoncg(problem *pr, Options *opts) {
    options = opts;
    prob = pr;
    l = opts->l;
    p = opts->p;
    batch_size = options->batch_size;
    cg_iters = options->cg_iters;
    eps = options->eps;
    
    step_alg = BACKTRACKING_LINE_SEARCH_BATCH; //opts->step_alg;// FIXED_EPOCH FIXED BACKTRACKING_LINE_SEARCH_BATCH, BACKTRACKING_LINE_SEARCH_FULL
    step_init = 0.1;
    
    // allocating memory to trcg method auxiliary variables.
    d = new double[p];
	Hd = new double[p];
    diag = new double[l];  // take care it's size will change.
    
    // newtoncg helper variables
    gradient = new double[p];
    s = new double[p];
    r = new double[p];
}

newtoncg::~newtoncg() {
    if(d!=NULL)
        delete[] d;
    if(Hd!=NULL)
        delete[] Hd;
    if(diag!=NULL)
        delete[] diag;
    
    if(gradient!=NULL)
        delete[] gradient;
    if(s!=NULL)
        delete[] s;
    if(r!=NULL)
        delete[] r;
    
}

void newtoncg::solve(double *wts, outputs *output) {
    // values for trust-region related values taken from LIBLINEAR.
    // Parameters for updating the iterates.
// 	double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;
	// Parameters for updating the trust region size delta.
// 	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;
    double alpha_pcg = 0.01;
    size_t cg_count;
    // exit tolerance
    double temp, gnorm=0, gnorm0;
    double initial_sample_size;
    double *costs, *time, *accuracy;
    size_t max_iters, iter, number_of_batches, start_index, end_index, cgs = 0;
    bool run_flag = true,  test_cost_flag = false, info = false, full = false;
    double alpha, f, step;
    double rate, exp_growth_constant;

    clock_t tic, toc, t1, t2; // variables to record the running time.
    
    double *w_new = new double[p];

    w = wts;    // initialize the w of ston class to use in other functions.
    max_iters = options->max_iters;
    costs = output->costs;
    time = output->time;
    accuracy = output->accuracy;

    // initialise the output
    iter = 0;
    time[iter] = 0;
//     grad_calc_count = 0;
    if(test_cost_flag) {
        f = prob->cost_test(w);
    } else {
        f = prob->cost_full(w);
    }
    costs[iter] = f;
    accuracy[iter] = prob->accuracy(w);
    prob->initilize_perm_idx();
    prob->gradient_full(w, gradient);
    gnorm0 = norm_(p, gradient);
    
    srand(2018);
    std::cout.setf(ios::fixed);
    std::cout<<"Newton-CG: "<<setw(3)<<iter<<" f:"<<setprecision(15)<< costs[iter]<<", accuracy:"<<setprecision(4)<<accuracy[iter]<<", Time:"<<setprecision(5)<< time[iter]<<std::endl;
   
    // some constants for the sample calculations
    initial_sample_size = 0.01;
    rate = 5;
    exp_growth_constant = pow((1.0-initial_sample_size)*l, 1.0/rate); // exponential
    
    // total gradient calculations during one epoch:
    output->grad_calc[0] = 3;
    
    // set start time
    tic = clock();
    
    while(iter < max_iters && run_flag) { // outer iterations
        iter = iter + 1;
        
        // linear batch sizes.
        batch_size = (size_t) ceil(min((double)l, max(l*initial_sample_size, (double)l/rate*iter)));
        number_of_batches = (size_t) floor((double)l/(batch_size));
//         // exponential batching scheme
//         batch_size = (size_t) ceil(min((double)l, l*initial_sample_size + pow(exp_growth_constant, iter)));
//         number_of_batches = (size_t) floor((double)l/(batch_size));
        // Randomizing the data points.
        if(number_of_batches!=1) {
            prob->randomize();
        }
        cgs = 0;
//         size_t loop_length = ceil(number_of_batches/2.0);
        for(size_t iter_batch=1; iter_batch<=number_of_batches; ++iter_batch) {
            start_index = (iter_batch-1) * batch_size;
            end_index = start_index + batch_size - 1;
//             start_index = 0; end_index = batch_size-1;
            if(iter_batch==number_of_batches) {
                    batch_size = l - batch_size*iter_batch + batch_size;
                    end_index = l-1;
            }
            // calculating the reduced gradient values.
            prob->gradient_batch(w, start_index, end_index, gradient);
            
            // solve the linear system using trust-region cg method
            cg_count = cg_batch(start_index, end_index);           
            cgs = cgs + cg_count;
//             std::cout<<"snorm"<<norm_(p, s)<<"\n";
            // updating the solution
            if(step_alg==BACKTRACKING_LINE_SEARCH_BATCH) {
                double rho = 0.5;
                double c = 1e-4;   // might need to adjust the value of c and change
                // stochastic backtracking line search method
                step = prob->backtracking_mb_ls(w, gradient, s, rho, c, start_index, end_index);
            } else if(step_alg==FIXED) {
                step = step_init;
            } else {
                step = step_init/(double)(iter);
            }
            axpy_(p, step, s, w);        
            
            if(info) {
                std::cout<<" f:"<<setprecision(10)<<f<<" |g|:"<<setprecision(10)<< gnorm<<"\n";
            }           
        }
        // store output for the iteration
        t1 = clock();
        if(test_cost_flag) {
            f = prob->cost_test(w);
        } else {       
            f = prob->cost_full(w);
        }
        costs[iter] = f;
        accuracy[iter] = prob->accuracy(w);
        t2 = clock();
        toc = clock();
        time[iter] = (toc - tic - iter*(t2-t1))/(double) CLOCKS_PER_SEC;
        
        gnorm = norm_(p, gradient);    
        // display the output after the iteration
        std::cout<<"Newton-CG:"<<setw(3)<<iter<<" f:"<<setprecision(15)<< costs[iter]<<" |g|:"<<setprecision(10)<< gnorm<<" Time:"<<setprecision(2)<< time[iter]<<" accuracy:"<<setprecision(4)<<accuracy[iter]<<" CG:"<<setw(3)<<cgs<<std::endl;
//         std::cout<<"gnorm: "<<gnorm<<" "<<eps<<" "<<gnorm0<<" "<<(gnorm <= eps*gnorm0)<<"\n";
        if (gnorm <= eps*gnorm0) {
            std::cout<<("reached required tolerance\n");
            run_flag = false;
//             break;
        }
    }
    
    // clear memory
    if(w_new!=NULL)
        delete[] w_new;
    std::cout<<("Newton-CG/++ completed...\n\n");
}

// This is used to solve the trust-region subproblem using the cg method with mini-batches
size_t newtoncg::cg_batch(size_t start_index, size_t end_index) {
    double tol = 0.1, cg_tol, norm_r, norm_r_new, alpha, beta;
    double std, sts, dtd, dsq, rad, norm_grad;
    size_t batch_size = end_index - start_index + 1;
    bool info = false;
    size_t k;

    // initialize
    size_t i;
    for(i=0; i<p; ++i) {
        s[i] = 0;
        r[i] = - gradient[i];
        d[i] = r[i];
    }
    norm_r = norm_(p, r);
    norm_grad = norm_(p, gradient);
    tol = tol * norm_grad;

    // calculating the diagonal matrix
    prob->get_diag(w, diag, start_index, end_index);
    k = 0;
    for(i=0; i< cg_iters; ++i) {  // inner cg iterations
        // check the residual condition
        if(norm_r <= tol) {
            break;
        }   // think of reducing number of params passed to functions calls
        k++;
        prob->Hessian_vector_product_batch(d, diag, Hd, start_index, end_index);
//         std::cout<<"Hd: "<<norm_(p, Hd)<<"\n";
        alpha = norm_r*norm_r / dot_(p, d, Hd);
        // update s
        axpy_(p, alpha, d, s);
//         std::cout<<"alpha: "<<alpha<<"\n";
        // update r
        alpha = -alpha;
        axpy_(p, alpha, Hd, r);
        norm_r_new = norm_(p, r);
        // calculate the betai
        beta = norm_r_new*norm_r_new/(norm_r*norm_r);
        norm_r = norm_r_new;
        // update di
        for(size_t j=0; j<p; ++j) {
            d[j] = r[j] + beta*d[j];
        }
    }
    if(info && (k==cg_iters)) {
         std::cout<<"max cg iters reached.\n";
    }
    return k;
}