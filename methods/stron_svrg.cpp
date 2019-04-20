/**
 *
 * Usage: This file defines the stron_svrg class for the STRON-SVRG in C++.
 *        It uses variance reduction using SVRG and progressive batching for Hessian calculations.
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

#include "stron_svrg.h"


stron_svrg::stron_svrg(problem *pr, Options *opts) {
    options = opts;
    prob = pr;
    l = opts->l;
    p = opts->p;
    batch_size = options->batch_size;
    cg_iters = options->cg_iters;

    // allocating memory to trcg method auxiliary variables.
    d = new double[p];
	Hd = new double[p];
    diag = new double[l];  // take care it's size will change.
    
    // stron_svrg helper variables
    gradient = new double[p];
    s = new double[p];
    r = new double[p];
}
stron_svrg::~stron_svrg() {
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
void stron_svrg::solve(double *wts, outputs *output) {
    // values for trust-region related values (Hsia et al 2018).
    // Parameters for updating the iterates.
	double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;
	// Parameters for updating the trust region size delta.
	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;
    double alpha_pcg = 0.01;
    size_t cg_count;
    // exit tolerance
    double eps = 1e-10, temp, gnorm0, gnorm;
    double rate, exp_growth_constant, initial_sample_size;
    
    double *costs, *time, *accuracy;
    size_t max_iters, iter, number_of_batches, start_index, end_index, cgs=0;
    bool run_flag = true,  test_cost_flag = false, info = false, full = true;
    double alpha, f_old, f_new, pred_red, actual_red, gTs, sTr, snorm;

    clock_t tic, toc; // variables to record the running time.
    
    double *full_grad = new double[p];
    double *grad_old = new double[p];
    
    double *w_new = new double[p];
    double *w0 = new double[p];

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

    srand(2018);
    std::cout.setf(ios::fixed);
    std::cout<<"STRON-SVRG:"<<setw(3)<<iter<<" f:"<<setprecision(15)<< costs[iter]<<", accuracy:"<<setprecision(4)<<accuracy[iter]<<", T:"<<setprecision(5)<< time[iter] <<std::endl;

    number_of_batches = (size_t) ceil((double)l/(batch_size));
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
        // linear batch sizes.
        batch_size_hessian = (size_t) ceil(min((double)l, max(l*initial_sample_size, (double)l/rate*iter)));
//         number_of_batches_hessian = (size_t) floor((double)l/(batch_size));
        // exponential batching scheme
//         batch_size_hessian = (size_t) ceil(min((double)l, l*initial_sample_size + pow(exp_growth_constant, iter)));
//         number_of_batches_hessian = (size_t) floor((double)l/(batch_size));
        
        // Randomizing the data points.
        prob->randomize();

        prob->gradient_full(w, full_grad);
        if(iter == 1) {
            gnorm0 = norm_(p, full_grad);
            delta = gnorm0;
        }        
        for(size_t i=0; i<p; ++i) { w0[i] = w[i]; }
        
        cgs = 0;
        for(size_t iter_batch=1; iter_batch<=number_of_batches; ++iter_batch) {
            start_index = (iter_batch-1) * batch_size;
            end_index = start_index + batch_size - 1;
            if(end_index>=l) {
                end_index = l-1;
            }
            // calculating the reduced gradient values.
            prob->gradient_batch(w, start_index, end_index, gradient);
            if(iter>1) {
                prob->gradient_batch(w0, start_index, end_index, grad_old); // this can be optimized                
                for(size_t i=0; i<p; ++i) {
                    gradient[i] = gradient[i] - grad_old[i] + full_grad[i];
                }
            }
            
            // solve the linear system using trust-region cg method
            cg_count = trcg_batch();
            
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
            snorm = norm_(p, s);
            if(iter == 1 && iter_batch == 1) {
                gnorm0 = norm_(p, gradient);
                delta = min(delta, snorm);
            }
            if(f_new - f_old - gTs <= 0) {
              alpha = sigma3;
            } else {
              alpha = max(sigma1, -0.5*(gTs/(f_new - f_old - gTs)));
            }
            if(actual_red < eta0*pred_red) {
                delta = min(alpha*snorm, sigma2*delta);
            } else if(actual_red < eta1*pred_red) {
              delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta));
            } else if(actual_red < eta2*pred_red) {
              delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta));
            } else {
              if(at_boundary) {
                  delta = sigma3*delta;
              } else {
                  delta = max(delta, min(alpha*snorm, sigma3*delta));
              }
            }

            // updating the solution
            if((actual_red > eta0 * pred_red)) {    // && (f_new < f_old)
                for(size_t i=0; i<p; ++i) {
                    w[i] = w_new[i];
                }
                gnorm = norm_(p, gradient);
                if (gnorm <= eps*gnorm0) {
                    std::cout<<("reached required tolerance\n");
                    run_flag = false;
                    break;
                }
                f_old = f_new;
            }
            if(info) {
                std::cout<<"act "<<setprecision(10)<<actual_red<<" pre "<<setprecision(10)<<pred_red<<" delta "<<setprecision(10)<<delta<<" f "<<setprecision(10)<<f_old<<" |g| "<<setprecision(10)<< gnorm<<" CG "<<setw(3)<<cg_count<<"\n";
            }
            if(f_old < -1.0e+32) {
              std::cout<<("WARNING: f < -1.0e+32\n");
              run_flag = false;
              break;
            }
            if ((abs(actual_red) <= 0) && (pred_red <= 0)) {
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
        std::cout<<"STRON-SVRG: iter:"<<setw(3)<<iter<<" act: "<<setprecision(10)<< actual_red<<" pre: "<<setprecision(10)<< pred_red<<" delta: "<<setprecision(10)<< delta<<" f: "<<setprecision(15)<< costs[iter]<<" |g|: "<<setprecision(10)<< gnorm<<" ACG "<<setw(3)<<int(ceil(cgs/(double)number_of_batches))<<" Time: "<<setprecision(2)<< time[iter]<<" accuracy: "<<setprecision(4)<<accuracy[iter] <<std::endl;
    }
    // clear memory
    if(full_grad!=NULL)
        delete[] full_grad;
    if(grad_old!=NULL)
        delete[] grad_old;
    if(w_new!=NULL)
        delete[] w_new;
    if(w0!=NULL)
        delete[] w0;
    std::cout<<("STRON-SVRG completed.\n");
}

// 
size_t stron_svrg::trcg_batch() {
    double tol = 0.1, cg_tol, norm_r, norm_r_new, alpha, beta;
    double std, sts, dtd, dsq, rad, norm_grad;
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
    at_boundary = false;

    // calculating the diagonal matrix
    prob->get_diag(w, diag, 0, batch_size_hessian-1);
    k = 0;
    for(i=0; i<cg_iters; ++i) {  // inner cg iterations
        // check the residual condition
        if(norm_r <= tol) {
            break;
        }   // think of reducing number of params passed to functions calls
        k++;
        prob->Hessian_vector_product_batch(d, diag, Hd, 0, batch_size_hessian-1);
        alpha = norm_r*norm_r / dot_(p, d, Hd);
        // update s
        axpy_(p, alpha, d, s);
        if(norm_(p, s) > delta) {
//             std::cout<< "in snrom "<<norm_(p, s)<<" in rnrom "<<norm_(p, r) <<" delta: "<<delta<<"\n";
            at_boundary = true;
            if(info) {
//                 std::cout<<"Hd: "<<norm_(p, Hd)<<"snorm: "<<norm_(p, s)<<" alpha: "<<alpha<<" delta: "<<delta<<"\n";
                std::cout<<("cg reached at trust region boundary\n");
            }
            alpha = -alpha;
            axpy_(p, alpha, d, s);
//             std::cout<< "in snrom "<<norm_(p, s)<<" in rnrom "<<norm_(p, r) <<" delta: "<<delta<<"\n";
            std = dot_(p, s, d);
            sts = dot_(p, s, s);
            dtd = dot_(p, d, d);
            dsq = delta*delta;
            rad = sqrt(std*std + dtd*(dsq-sts));
            if(std >= 0) {
                alpha = (dsq - sts)/(std + rad);
            } else {
                alpha = (rad - std)/dtd;
            }
            axpy_(p, alpha, d, s);
            alpha = -alpha;
            axpy_(p, alpha, Hd, r);
//             std::cout<< "in snrom "<<norm_(p, s)<<" in rnrom "<<norm_(p, r) <<" delta: "<<delta<<"\n";
            break;
        }
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
//         std::cout<< "out snrom "<<norm_(p, s)<<" delta: "<<delta<<"\n";
    }
    if(info && (k==cg_iters)){
         std::cout<<"max cg iters reached.\n";
    }

    return k;
}