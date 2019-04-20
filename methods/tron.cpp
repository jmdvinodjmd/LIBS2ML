/**
 *
 * Usage: This file defines the tron class for the TRON method in C++ (Hsia et al. 2018).
﻿@InProceedings{Hsia2018,
	title = 	 {Preconditioned Conjugate Gradient Methods in Truncated Newton Frameworks for Large-scale Linear Classification},
	author = 	 {Chih-Yang Hsia and Wei-Lin Chiang and Chih-Jen Lin},
	booktitle = 	 {ACML},
	year = 	 {2018},
	publisher = 	 {PMLR}
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

#include "tron.h"

tron::tron(problem *pr, Options *opts) {
    options = opts;
    prob = pr;
    l = opts->l;
    p = opts->p;
    cg_iters = options->cg_iters;
    eps = options->eps;
    d = new double[p];
	Hd = new double[p];
	z = new double[p];
    
    gradient = new double[p];
    s = new double[p];
    r = new double[p];
}

tron::~tron() {
    if(d!=NULL)
        delete[] d;
    if(Hd!=NULL)
        delete[] Hd;
    if(z!=NULL)
        delete[] z;
    
    if(gradient!=NULL)
        delete[] gradient;
    if(s!=NULL)
        delete[] s;
    if(r!=NULL)
        delete[] r;
}

// The learning algorithm goes here
void tron::solve(double *w, outputs *output) {
    // values for trust-region related values
    // Parameters for updating the iterates.
	double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;
	// Parameters for updating the trust region size delta.
	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;
    double alpha_pcg = 0.01;
    size_t cg_count;

    double temp, delta, gnorm0, gnorm;
    
    double *costs, *time, *accuracy;
    size_t max_iters, iter;
    bool run_flag = true, test_cost_flag = false, at_boundary;
    double alpha, f_old, f_new, pred_red, actual_red, gTs, sTr, sTpre_cond_Hnorm;
    
    clock_t tic, toc, t1, t2; // variables to record the running time.
    double *diag = new double[l];
    double *pre_cond_H = new double[p];
    
    double *w_new = new double[p];
    
    max_iters = options->max_iters;
    costs = output->costs;
    time = output->time;
    accuracy = output->accuracy;
    
    // initialise the output
    iter = 0;
    time[iter] = 0;
//     grad_calc_count = 0;
    if(test_cost_flag) {
        f_old = prob->cost_test(w);
    } else {
        f_old = prob->cost_full(w);
    }
    costs[iter] = f_old;
    accuracy[iter] = prob->accuracy(w);
    srand(2018);
    std::cout.setf(ios::fixed);
    std::cout<<"TRON: Iter = "<<setw(3)<<iter<<", cost: "<<setprecision(10)<< costs[iter]<<", time: "<<setprecision(10)<< time[iter] <<", accuracy: "<<setprecision(10)<<accuracy[iter]<<std::endl;
    
    // total gradient calculations during one epoch:
    output->grad_calc[0] = 1;
  
    // set start time
    tic = clock();    
    while(iter < max_iters && run_flag) { // outer iterations
        iter = iter + 1;        
        prob->gradient_full(w, gradient); 
        
        // calculating the diagonal matrix and diagonal preconditioner
        prob->get_diag_pre_condH(w, diag, pre_cond_H);        
        for(size_t i=0; i<p; i++) {
            pre_cond_H[i] = (1-alpha_pcg) + alpha_pcg*pre_cond_H[i];
        }
        temp = uTMv(gradient, pre_cond_H, gradient);
        
        if(iter == 1) {
            delta = sqrt(temp);           
        }

        // solve the linear system using trust-region pcg method
        cg_count = trpcg(pre_cond_H, diag, delta, cg_iters, &at_boundary);        
        for(size_t i=0; i<p; ++i) {
            w_new[i] = w[i] + s[i]; 
        }        
        f_new = prob->cost_full(w_new);        
        actual_red = f_old - f_new;
        gTs = dot_(p, gradient, s);        
        sTr = dot_(p, s, r);        
        pred_red = -0.5 * ( gTs - sTr);
        
        // updating the trust-region radius
        sTpre_cond_Hnorm = sqrt(uTMv(s, pre_cond_H, s));
        if(iter == 1) { 
            gnorm0 = norm_(p, gradient);
            delta = min(delta, sTpre_cond_Hnorm);
        }
        if(f_new - f_old - gTs <= 0) {
          alpha = sigma3;
        } else {
          alpha = max(sigma1, -0.5*(gTs/(f_new - f_old - gTs)));
        }
        if(actual_red < eta0*pred_red) {
            delta = min(alpha*sTpre_cond_Hnorm, sigma2*delta);
        } else if(actual_red < eta1*pred_red) {
          delta = max(sigma1*delta, min(alpha*sTpre_cond_Hnorm, sigma2*delta));
        } else if(actual_red < eta2*pred_red) {
          delta = max(sigma1*delta, min(alpha*sTpre_cond_Hnorm, sigma3*delta));
        } else {
          if(at_boundary) {
              delta = sigma3*delta;
          } else {
              delta = max(delta, min(alpha*sTpre_cond_Hnorm, sigma3*delta));
          }
        }
    
        // updating the solution
        if(actual_red > eta0 * pred_red) {
            axpy_(p, 1.0, s, w);
            gnorm = norm_(p, gradient);            
            if (gnorm <= eps*gnorm0) {
                std::cout<<("reached required tolerance\n");
                run_flag = false;
                break;
            }
            f_old = f_new;
        }
        
        // store output for the iteration
        t1 = clock();
        if(test_cost_flag) {
            f_old = prob->cost_test(w);
        }
        costs[iter] = f_old;
        accuracy[iter] = prob->accuracy(w);
        t2 = clock();
        toc = clock();
        time[iter] = (toc - tic - iter*(t2-t1))/(double) CLOCKS_PER_SEC;      
        std::cout<<"TRON iter:"<<setw(3)<<iter<<" act: "<<setprecision(10)<<actual_red<<" pre: "<<setprecision(10)<<pred_red<<" delta: "<<setprecision(7)<<delta<<" f: "<<setprecision(12)<<costs[iter]<<" |g|: "<<setprecision(10)<< gnorm<<" CG "<<setw(2)<<cg_count<<", time: "<<setprecision(5)<< time[iter] <<", accuracy: "<<setprecision(4)<<accuracy[iter]<<std::endl; 
        
        // exit if tolerance reached
        if(f_old < -1.0e+32) {
          std::cout<<("WARNING: f < -1.0e+32\n");
          run_flag = false;
          break;
        }
        if((abs(actual_red) <= 0) && (pred_red <= 0)) {
          std::cout<<("WARNING: actred and prered <= 0\n");
          run_flag = false;
          break;
        }
        if(abs(actual_red) <= 1.0e-12*abs(f_old) && abs(pred_red) <= 1.0e-12*abs(f_old)) {
          std::cout<<("WARNING: actred and prered too small\n");
          run_flag = false;
          break;
        }
    }
    
    // clear memory
    if(w_new!=NULL)
        delete[] w_new;
    if(pre_cond_H!=NULL)
        delete[] pre_cond_H;
    if(diag!=NULL)
        delete[] diag;
//     std::cout<<("TRON completed.\n");
}


// This is used to solve the trust-region subproblem using the pcg method.
size_t tron::trpcg(double *pre_cond_H, double *diag, 
            double delta, size_t cg_iters, bool *at_boundary) {
    
    double tol = 0.1, cg_tol, zTr, znewTrnew, alpha, beta;
    double sTMd, sTMs, dTMd, dsq, rad, sTpre_cond_Hnorm;
    
    size_t k;
    
    // initialize the vectors
    for(size_t i=0; i<p; ++i) {
        s[i] = 0;
        r[i] = - gradient[i];
        z[i] = r[i] / pre_cond_H[i];
        d[i] = z[i];
    }
    zTr = dot_(p, z, r);
    
    cg_tol = tol*sqrt(zTr);
    *at_boundary = false;
    k = 0;
    for(size_t i=0; i<cg_iters; ++i) { // inner cg iterations
        // check the residual condition        
        if(sqrt(zTr) <= cg_tol) {
            break;
        }
        k = k + 1;
        // calculate the alpha
        prob->Hessian_vector_product(d, diag, Hd);        
        alpha = zTr/dot_(p, d, Hd);
        // udate s
        axpy_(p, alpha, d, s);
        
        sTpre_cond_Hnorm = sqrt(uTMv(s, pre_cond_H, s));
        if(sTpre_cond_Hnorm > delta) {
            std::cout<<("cg reached at trust region boundary\n");
            *at_boundary = true;
            alpha = -alpha;
            axpy_(p, alpha, d, s);
            sTMd = uTMv(s, pre_cond_H, d);
            sTMs = uTMv(s, pre_cond_H, s);
            dTMd = uTMv(d, pre_cond_H, d);     
            dsq = delta*delta;
            rad = sqrt(sTMd*sTMd + dTMd*(dsq-sTMs));
            if(sTMd >= 0) {
                alpha = (dsq - sTMs)/(sTMd + rad);
            } else {
                alpha = (rad - sTMd)/dTMd;
            }
            axpy_(p, alpha, d, s);
            alpha = -alpha;
            axpy_(p, alpha, Hd, r);
            
            break;
        }
        // update r and z
        alpha = -alpha;
        axpy_(p, alpha, Hd, r);
        for(size_t j=0; j<p; ++j) {
            z[j] = r[j]/pre_cond_H[j];
        }
        znewTrnew = dot_(p, z, r);
        beta = znewTrnew/zTr;
        zTr = znewTrnew;
        // update d
        for(size_t j=0; j<p; ++j) {
            d[j] = z[j] + beta*d[j];
        }        
//         std::cout<< "snorm"<<norm(p, s)<< "rnorm"<<norm(p, r)<< "dnorm"<<norm(p, d)<<"\n";
    }
    if(k==cg_iters) {
        std::cout<<("max cg iters reached.\n");
    } 
    return k;
}

double tron::uTMv(double *u, double *M, double *v) {
    double temp = 0;
    for(size_t i=0; i<p; i++) {
        temp += M[i]*u[i]*v[i];
    }
    return temp;
}