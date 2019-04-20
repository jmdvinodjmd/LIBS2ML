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

#include "lbfgs.h"

lbfgs::lbfgs(problem *pr, Options *opts) {
    options = opts;
    prob = pr;
    l = opts->l;
    p = opts->p;
    L = 5;
    step_alg = BACKTRACKING_LINE_SEARCH_FULL; //opts->step_alg;// FIXED_EPOCH FIXED BACKTRACKING_LINE_SEARCH_FULL, BACKTRACKING_LINE_SEARCH_FULL
    step_init = 0.001;
    
    s_array = new double*[L];
    for(size_t i=0; i<L; ++i) {
        s_array[i] = new double[p];
    }
	y_array = new double*[L];
    for(size_t i=0; i<L; ++i) {
        y_array[i] = new double[p];
    }
    
    q = new double[p];
    a = new double[L];
    rk = new double[L];
    R = new double[p];
}
lbfgs::~lbfgs() {     
    
    if(s_array!=NULL) {
        for(size_t i=0; i<L; ++i) {
            delete[] s_array[i];
        }
        delete[] s_array;
    }
    if(y_array!=NULL) {
        for(size_t i=0; i<L; ++i) {
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

void lbfgs::solve(double *wts, outputs *output) {
    double *costs, *time, *accuracy;
    size_t max_iters, iter;
    bool info=false, test_cost_flag=false;
    double step, f_val;

    clock_t tic, toc; // variables to record the running time.
    double *full_grad = new double[p];
    double *full_grad_new = new double[p];
    double *Hg = new double[p];
//     double *Hv = NULL;
//         Hv = new double[p];
    
    double *w0 = new double[p];

    w = wts;    // initialize the w of ston class to use in other functions.
    max_iters = options->max_iters;
    costs = output->costs;
    time = output->time;
    accuracy = output->accuracy;
    
    // initialise the output
    iter = 0;
    time[iter] = 0;
    if(test_cost_flag) {
        f_val = prob->cost_test(w);
    } else {
        f_val = prob->cost_full(w);
    }
    costs[iter] = f_val;
    accuracy[iter] = prob->accuracy(w);
    
    srand(2018);
    for(size_t i=0; i<p; ++i) { w0[i] = w[i]; }
    prob->gradient_full(w, full_grad_new);
    
    std::cout.setf(ios::fixed);
    std::cout<<"LBFGS: iter:"<<setw(3)<<iter<<" f: "<<setprecision(15)<< costs[iter]<<", accuracy: "<<setprecision(4)<<accuracy[iter]<<", Time: "<<setprecision(5)<< time[iter] <<std::endl;

    // set start time
    tic = clock();
    
    while(iter < max_iters) { // outer iterations              
        if(iter > 0) {
            for(size_t i=0; i<p; ++i) { full_grad[i] = full_grad_new[i]; }
            prob->gradient_full(w, full_grad_new);        
            // store curvature pair
            storeCurvature(w, w0, full_grad_new, full_grad);
//             storeCurvature(u_new, u_old, Hv);            
            for(size_t i=0; i<p; ++i) { w0[i] = w[i]; }            
        }
        double c = 1e-4;   // might need to adjust the value of c and change
        if(iter > 0) {
            // perform LBFGS two loop recursion
            lbfgs_two_loop_recursion(full_grad_new, Hg);
            if(step_alg==BACKTRACKING_LINE_SEARCH_FULL) {
                double rho = 0.5;
//                 double c = 1e-4;   // might need to adjust the value of c and change
                step = prob->backtracking_ls(w, full_grad_new, Hg, rho, c);
            } else 
                if(step_alg==FIXED) {
                step = step_init;
            } else {
                step = step_init/(double)(iter+1);
            }
            axpy_(p, step, Hg, w);
        } else {
            if(step_alg==BACKTRACKING_LINE_SEARCH_FULL) {
                // using the backtracking line search
                double rho = 0.5;
//                 double c = 1e-4;   // might need to adjust the value of c and change
                // the backtracking algorithm.
                // stochastic backtracking line search method
                step = prob->backtracking_ls(w, full_grad_new, w, rho, c);
            } else 
                if(step_alg==FIXED) {
                step = step_init;
            } else if(FIXED_EPOCH) {
                step = step_init/(double)(iter+1);
            }
            axpy_(p, -step, full_grad_new, w);
        }

        iter = iter + 1;
        // measure elapsed time
        if(test_cost_flag) {
            f_val = prob->cost_test(w);
        } else {
            f_val = prob->cost_full(w);
        }
        costs[iter] = f_val;
        accuracy[iter] = prob->accuracy(w);
        toc = clock();
        time[iter] = (toc - tic)/(double) CLOCKS_PER_SEC;

        // display the output after the iteration
        std::cout<<"LBFGS: iter:"<<setw(3)<<iter<<" f: "<<setprecision(15)<< costs[iter]<<" |g|: "<<setprecision(10)<< norm_(p, full_grad_new)<<" Time: "<<setprecision(2)<< time[iter]<<" accuracy: "<<setprecision(4)<<accuracy[iter] <<std::endl;
    }
            
    // clear memory
    if(full_grad!=NULL)
        delete[] full_grad;
    if(full_grad_new!=NULL)
        delete[] full_grad_new;
    if(Hg!=NULL)
        delete[] Hg;
//     if(Hv!=NULL)
//         delete[] Hv;        
    if(w0!=NULL)
        delete[] w0;
    
    std::cout<<("LBFGS/c++ completed...\n\n");
}

void lbfgs::lbfgs_two_loop_recursion(double *grad, double *HessGrad) {
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

void lbfgs::storeCurvature(double *w, double *w0, double *grad_new, double *grad_old) {
    if(memory_size!=L) {
        memory_size++;
    }
    insert_at = (++insert_at) % L;
    for(size_t i=0; i<p; ++i) {
        s_array[insert_at][i] = w[i]-w0[i];
        y_array[insert_at][i] = grad_new[i]-grad_old[i];
    }
}

void lbfgs::storeCurvature(double *w, double *w0, double *Hv) {
    if(memory_size!=L) {
        memory_size++;
    }
    insert_at = (++insert_at) % L;
    for(size_t i=0; i<p; ++i) {
        s_array[insert_at][i] = w[i]-w0[i];
        y_array[insert_at][i] = Hv[i];
    }
}