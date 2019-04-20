/**
 *
 * Usage: This file defines the stron_pcg_prog class for the STRON-PCG-PROG method (Chauhan et al 2018) in C++.
 *          It uses STRON with pcg subproblem solver and progressive subsampling.
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

#include "stron_pcg_prog.h"

stron_pcg_prog::stron_pcg_prog(problem *pr, Options *opts) {
    options = opts;
    prob = pr;
    l = opts->l;
    p = opts->p;
    batch_size = options->batch_size;
    cg_iters = options->cg_iters;

    // allocating memory to trcg method auxiliary variables.
    d = new double[p];
	Hd = new double[p];
    z = new double[p];
    diag = new double[l];  // take care it's size will change.
    pre_cond_H = new double[p];
    
    // stron_pcg_prog helper variables
    gradient = new double[p];
    s = new double[p];
    r = new double[p];
}
stron_pcg_prog::~stron_pcg_prog() {
    if(d!=NULL)
        delete[] d;
    if(Hd!=NULL)
        delete[] Hd;
    if(diag!=NULL)
        delete[] diag;
    if(z!=NULL)
        delete[] z;
    if(pre_cond_H!=NULL)
        delete[] pre_cond_H;
    
    if(gradient!=NULL)
        delete[] gradient;
    if(s!=NULL)
        delete[] s;
    if(r!=NULL)
        delete[] r;
    
}
void stron_pcg_prog::solve(double *wts, outputs *output) {
    // values for trust-region related values taken from LIBLINEAR.
    // Parameters for updating the iterates.
	double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;
	// Parameters for updating the trust region size delta.
	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;
    double alpha_pcg = 0.01;
    size_t cg_count;
    // exit tolerance
    double eps = 1e-10, temp, gnorm0, gnorm;

    double *costs, *time, *accuracy;
    size_t max_iters, iter, number_of_batches, start_index, end_index, cgs=0;
    bool run_flag = true,  test_cost_flag = false, info = false, full = true;
    double alpha, f_old, f_new, pred_red, actual_red, gTs, sTr, sTpre_cond_Hnorm;

    clock_t tic, toc; // variables to record the running time.
 
    double *w_new = new double[p];
    double rate, exp_growth_constant, initial_sample_size;

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
        f_old = prob->cost_test(w);
    } else {
        f_old = prob->cost_full(w);
    }
    costs[iter] = f_old;
    accuracy[iter] = prob->accuracy(w);
    prob->initilize_perm_idx();
    prob->gradient_full(w, gradient);
    gnorm0 = norm_(p, gradient);
    delta = gnorm0;

    srand(2018);
    std::cout.setf(ios::fixed);
    std::cout<<"STRON_PCG_PROG: iter:"<<setw(3)<<iter<<" f: "<<setprecision(15)<< costs[iter]<<", accuracy: "<<setprecision(4)<<accuracy[iter]<<", Time: "<<setprecision(5)<< time[iter] <<std::endl;

    // some constants for the sample calculations
    initial_sample_size = 0.01;
    rate = 5;
    exp_growth_constant = pow((1.0-initial_sample_size)*l,1.0/rate); // exponential   
    
    // total gradient calculations during one epoch:
    output->grad_calc[0] = 3;
    // set start time
    tic = clock();
    
    while(iter < max_iters && run_flag) { // outer iterations
        iter = iter + 1;
//         // linear batch sizes.
        batch_size = (size_t) ceil(min((double)l, max(l*initial_sample_size, (double)l/rate*iter)));
        number_of_batches = (size_t) floor((double)l/(batch_size));
        // exponential batching scheme
//         batch_size = (size_t) ceil(min((double)l, l*initial_sample_size + pow(exp_growth_constant, iter)));
//         number_of_batches = (size_t) floor((double)l/(batch_size));
        // Randomizing the data points.
        if(number_of_batches!=1) {
            prob->randomize();
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
                cg_count = trpcg_batch(start_index, end_index);

                cgs = cgs + cg_count;
                for(size_t i=0; i<p; ++i) {
                    w_new[i] = w[i] + s[i];
                }
                if(full) {
                    f_new = prob->cost_full(w_new);
                } else
                {
                    f_old = prob->cost_batch(w, start_index, end_index);
                    f_new = prob->cost_batch(w_new, start_index, end_index);
                }

                actual_red = f_old - f_new;
                gTs = dot_(p, gradient, s);

                sTr = dot_(p, s, r);

                pred_red = -0.5 * ( gTs - sTr);

                // updating the trust-region radius
                sTpre_cond_Hnorm = sqrt(uTMv(s, pre_cond_H, s));
                if(iter == 1 && iter_batch == 1) { 
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
                if(actual_red > eta0 * pred_red) { // && (f_new < f_old)
                    axpy_(p, 1.0, s, w);
                    gnorm = norm_(p, gradient);            
                    if (gnorm <= eps*gnorm0) {
                        std::cout<<("reached required tolerance\n");
                        run_flag = false;
                        break;
                    }
                    f_old = f_new;
                }                
            }
        } else {
            // calculating the gradient value.
            prob->gradient_full(w, gradient);

            // solve the linear system using trust-region cg method
            cgs = trpcg();
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
            if(actual_red > eta0 * pred_red) { // && (f_new < f_old)
                axpy_(p, 1.0, s, w);
                gnorm = norm_(p, gradient);            
                if (gnorm <= eps*gnorm0) {
                    std::cout<<("reached required tolerance\n");
                    run_flag = false;
                    break;
                }
                f_old = f_new;
            }
        }
        // store output for the iteration
        if(test_cost_flag) {
            f_old = prob->cost_test(w);
        } else if(!full) {           
            f_old = prob->cost_full(w);
        }
        costs[iter] = f_old;
        accuracy[iter] = prob->accuracy(w);
        toc = clock();
        time[iter] = (toc - tic)/(double) CLOCKS_PER_SEC;

        // display the output after the iteration
        std::cout<<"STRON_PCG_PROG: iter:"<<setw(3)<<iter<<" act: "<<setprecision(10)<< actual_red<<" pre: "<<setprecision(10)<< pred_red<<" delta: "<<setprecision(10)<< delta<<" f: "<<setprecision(15)<< costs[iter]<<" |g|: "<<setprecision(10)<< gnorm<<" CG "<<setw(3)<<cgs<<" Time: "<<setprecision(2)<< time[iter]<<" accuracy: "<<setprecision(4)<<accuracy[iter] <<std::endl;
        if(f_old < -1.0e+32) {
          std::cout<<("WARNING: f < -1.0e+32\n");
          run_flag = false;
//           break;
        }
        if ((abs(actual_red) <= 0) && (pred_red <= 0)) {
          std::cout<<("WARNING: actred and prered <= 0\n");
          run_flag = false;
//           break;
        }
        if(abs(actual_red) <= 1.0e-12*abs(f_old) && abs(pred_red) <= 1.0e-12*abs(f_old)) {
          std::cout<<("WARNING: actred and prered too small\n");
          run_flag = false;
//           break;
        }
    }
    // clear memory
    if(w_new!=NULL)
        delete[] w_new;
    std::cout<<("STRON/++ completed...\n\n");
}

/** trpcg_batch()
 * This method is used to solve the trust-region subproble using the
 * cg method.
 */
size_t stron_pcg_prog::trpcg_batch(size_t start_index, size_t end_index) {
    double tol = 0.1, cg_tol, zTr, znewTrnew, alpha, beta;
    double sTMd, sTMs, dTMd, dsq, rad, sTpre_cond_Hnorm;
    double alpha_pcg = 0.01;
    bool info = false;
    size_t k;
    
    // calculating the diagonal matrix and diagonal preconditioner
    prob->get_diag_pre_condH_batch(w, diag, pre_cond_H, start_index, end_index);
    for(size_t i=0; i<p; i++) {
        pre_cond_H[i] = (1-alpha_pcg) + alpha_pcg*pre_cond_H[i];
    }
    
    // initialize
    size_t i;
    for(i=0; i<p; ++i) {
        s[i] = 0;
        r[i] = - gradient[i];
        z[i] = r[i] / pre_cond_H[i];
        d[i] = z[i];
    }
    zTr = dot_(p, z, r);
    cg_tol = tol*sqrt(zTr);
    at_boundary = false;

    k = 0;
    for(i=0; i<cg_iters; ++i) {  // inner cg iterations
        // check the residual condition
        // check the residual condition        
        if(sqrt(zTr) <= cg_tol) {
            break;
        }
        // think of reducing number of params passed to functions calls
        k++;
        prob->Hessian_vector_product_batch(d, diag, Hd, start_index, end_index);
        alpha = zTr/dot_(p, d, Hd);
        // update s
        axpy_(p, alpha, d, s);
        sTpre_cond_Hnorm = sqrt(uTMv(s, pre_cond_H, s));
        if(sTpre_cond_Hnorm > delta) {
            at_boundary = true;
            if(info) {
                std::cout<<("cg reached at trust region boundary\n");
            }
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
        // update r
        alpha = -alpha;
        axpy_(p, alpha, Hd, r);
        for(size_t j=0; j<p; ++j) {
            z[j] = r[j]/pre_cond_H[j];
        }
        znewTrnew = dot_(p, z, r);
        beta = znewTrnew/zTr;
        zTr = znewTrnew;
        for(size_t j=0; j<p; ++j) {
            d[j] = z[j] + beta*d[j];
        }
    }
    if(info && (k==cg_iters)){
         std::cout<<"max cg iters reached.\n";
    }

    return k;
}


size_t stron_pcg_prog::trpcg() {
    double tol = 0.1, cg_tol, zTr, znewTrnew, alpha, beta;
    double sTMd, sTMs, dTMd, dsq, rad, sTpre_cond_Hnorm;
    double alpha_pcg = 0.01;
    bool info = false;
    size_t k;
    
    // calculating the diagonal matrix and diagonal preconditioner
    prob->get_diag_pre_condH(w, diag, pre_cond_H);
    for(size_t i=0; i<p; i++) {
        pre_cond_H[i] = (1-alpha_pcg) + alpha_pcg*pre_cond_H[i];
    }
    
    // initialize
    size_t i;
    for(i=0; i<p; ++i) {
        s[i] = 0;
        r[i] = - gradient[i];
        z[i] = r[i] / pre_cond_H[i];
        d[i] = z[i];
    }
    zTr = dot_(p, z, r);
    cg_tol = tol*sqrt(zTr);
    at_boundary = false;

    k = 0;
    for(i=0; i<cg_iters; ++i) {  // inner cg iterations
        // check the residual condition
        // check the residual condition        
        if(sqrt(zTr) <= cg_tol) {
            break;
        }
        // think of reducing number of params passed to functions calls
        k++;
        prob->Hessian_vector_product(d, diag, Hd);
        alpha = zTr/dot_(p, d, Hd);
        // update s
        axpy_(p, alpha, d, s);
        sTpre_cond_Hnorm = sqrt(uTMv(s, pre_cond_H, s));
        if(sTpre_cond_Hnorm > delta) {
            at_boundary = true;
            if(info) {
                std::cout<<("cg reached at trust region boundary\n");
            }
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
        // update r
        alpha = -alpha;
        axpy_(p, alpha, Hd, r);
        for(size_t j=0; j<p; ++j) {
            z[j] = r[j]/pre_cond_H[j];
        }
        znewTrnew = dot_(p, z, r);
        beta = znewTrnew/zTr;
        zTr = znewTrnew;
        for(size_t j=0; j<p; ++j) {
            d[j] = z[j] + beta*d[j];
        }
    }
    if(info && (k==cg_iters)){
         std::cout<<"max cg iters reached.\n";
    }

    return k;
}

double stron_pcg_prog::uTMv(double *u, double *M, double *v) {
    double temp = 0;
    for(size_t i=0; i<p; i++) {
        temp += M[i]*u[i]*v[i];
    }
    return temp;
}