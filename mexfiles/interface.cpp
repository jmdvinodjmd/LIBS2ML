/**
 *
 * Usage: This MEX file serves as an interface between MATLAB and C++.
 * Input:
 *          Data and parameters from the MATLAB.
 * Output:
 *          Returns required data in MATLAB format.
 * Functionality:
 *      - Take input from MATLAB and Convert input to LIBSVM format
 *      - Create objects of problem and solver classes
 *      - Call the solver and pass the data and paramters to solver (in C++)
 *      - Receive the results after learning from the solver and pass back to MATLAB 
 *
 * This file is part of LIBS2ML.
 * Created by V. K. Chauhan on Sept. 15, 2018
 * Modified by V. K. Chauhan on Apr. 14, 2019
 *
**/

// include standard libraries + classes for all problems and solvers.
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include "linear_svm.h"
#include "logistic_regression.h"
#include "stron.h"
#include "stron_pcg_prog.h"
#include "stron_pcg.h"
#include "stron_svrg.h"
#include "tron.h"
#include "sqn.h"
#include "NewtonCG.h"
#include "lbfgs.h"
#include "mex.h"
using namespace std;

// some global variables
struct inputs input;		// set by read_problem
struct feature_node *x_space = NULL;
int col_format_flag = true; // need to fix this part.
double bias = 0;

// Some code adapted from LIBLINEAR to keep the data formats same
static void fake_answer(int nlhs, mxArray *plhs[]) {
	int i;
	for(i=0; i<nlhs; i++)
		plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
}
int read_problem_sparse(const mxArray *instance_mat, const mxArray *label_vec,
        struct feature_node** xx, double *yy, size_t l) {
	mwIndex *ir, *jc, low, high, k;
	// using size_t due to the output type of matlab functions
	size_t i, j, elements, max_index, label_vector_row_num;
	mwSize num_samples;
	double *samples, *labels;
	mxArray *instance_mat_col; // instance sparse matrix in column format
	x_space = NULL;
    
	if(col_format_flag)
		instance_mat_col = (mxArray *)instance_mat;
	else {
		// transpose instance matrix
		mxArray *prhs[1], *plhs[1];
		prhs[0] = mxDuplicateArray(instance_mat);
		if(mexCallMATLAB(1, plhs, 1, prhs, "transpose")) {
			mexPrintf("Error: cannot transpose training instance matrix\n");
			return -1;
		}
		instance_mat_col = plhs[0];
		mxDestroyArray(prhs[0]);
	}
	// the number of instances
	label_vector_row_num = mxGetM(label_vec);
	if(label_vector_row_num!=l) {
		mexPrintf("Length of label vector does not match # of instances.\n");
		return -1;
	}

	// each column is one instance
	labels = mxGetPr(label_vec);
	samples = mxGetPr(instance_mat_col);
	ir = mxGetIr(instance_mat_col);
	jc = mxGetJc(instance_mat_col);

	num_samples = mxGetNzmax(instance_mat_col);
	elements = num_samples + l*2;
	max_index = mxGetM(instance_mat_col);

	x_space = Malloc(struct feature_node, elements);
	input.bias = bias;

	j = 0;
	for(i=0; i<l; i++) {
		xx[i] = &x_space[j];
		yy[i] = labels[i];
		low = jc[i], high = jc[i+1];
		for(k=low; k<high; k++) {
			x_space[j].index = (int) ir[k]+1;
			x_space[j].value = samples[k];
			j++;
	 	}
		if(input.bias>=0) {
			x_space[j].index = (int) max_index+1;
			x_space[j].value = input.bias;
			j++;
		}
		x_space[j++].index = -1;
	}

	if(input.bias>=0) // this step can be problematic for bias > 0
		input.p = (size_t) max_index+1;
	else
		input.p = (size_t) max_index;
	return 0;
}

// The following function is used to clear the memory for parameters.
void clear_params_memory(inputs *in) {
    if (in->x!=NULL) {
        free(in->x);
    }
    if (in->y!=NULL) {
        free(in->y);
    }
    if (in->x_test!=NULL) {
        free(in->x_test);
    }
    if (in->y_test!=NULL) {
        free(in->y_test);
    }
}

// This is the entry point for the interface.cpp
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // declaring input variables
    double *w;  // to be changed in place.
    struct Options options;

    // declaring output variables
    outputs output;
    size_t output_length;

    // declaring auxiliary variables
    int problem_type;
    char method_name[20], problem_name[25];
    size_t number_of_batches;

    // step 1. Receive input and parameter options from MATLAB.
    w = mxGetPr(prhs[0]);
    options.lambda = (double) mxGetScalar(prhs[5]);   // Penalty parameter
    options.batch_size = (size_t) mxGetScalar(prhs[6]);
    options.max_iters = (size_t) mxGetScalar(prhs[7]);
    mxGetString(prhs[8], method_name, mxGetN(prhs[8])+1);
    mxGetString(prhs[9], problem_name, mxGetN(prhs[9])+1);
    
    options.l = (size_t) mxGetN(prhs[1]);            // No. of data points      
    options.l_test = (size_t) mxGetN(prhs[2]);            // No. of data points.
    options.step_alg = FIXED;   // default value set. actual value controlled from solver.
    // selecting the solver type
	if(strcmp(method_name, "TRON") == 0) {
        options.method_type = TRON;
        output_length = options.max_iters+1;
    } else if (strcmp(method_name, "STRON") == 0) {
        options.method_type = STRON;
//         number_of_batches = (size_t) ceil((double)options.l/(options.batch_size));
//         output_length = options.max_iters * number_of_batches + 1;
        output_length = options.max_iters+1;
    } else if (strcmp(method_name, "SQN") == 0) {
        options.method_type = SQN;
        output_length = options.max_iters+1;
    } else if (strcmp(method_name, "SVRG_LBFGS") == 0) {
        options.method_type = SVRG_LBFGS;
        output_length = options.max_iters+1;
    } else if (strcmp(method_name, "SVRG_SQN") == 0) {
        options.method_type = SVRG_SQN;
        output_length = options.max_iters+1;
    } else if (strcmp(method_name, "LBFGS") == 0) {
        options.method_type = LBFGS;
        output_length = options.max_iters+1;
    } else if (strcmp(method_name, "NEWTON_CG") == 0) {
        options.method_type = NEWTON_CG;
        output_length = options.max_iters+1;
    } else if (strcmp(method_name, "STRON_SVRG") == 0) {
        options.method_type = STRON_SVRG;
        output_length = options.max_iters+1;
    } else if (strcmp(method_name, "STRON_PCG") == 0) {
        options.method_type = STRON_PCG;
        output_length = options.max_iters+1;
    } else if (strcmp(method_name, "STRON_PCG_PROG") == 0) {
        options.method_type = STRON_PCG_PROG;
        output_length = options.max_iters+1;
    } else {
        mexPrintf("Error: Non-existent solver.");
    }
    // selecting the problem type
    if(strcmp(problem_name, "LOGISTIC_REGRESSION_L2R") == 0) {
        problem_type = LOGISTIC_REGRESSION_L2R;
    } else if (strcmp(problem_name, "SVM_L2R_L2L") == 0) {
        problem_type = SVM_L2R_L2L;
    } else {
        mexPrintf("Error: Non-existent problem type.");
    }
    if((options.method_type==STRON) || (options.method_type==TRON)) {
        options.cg_iters = (size_t) mxGetScalar(prhs[10]);
    }
    options.eps = 1e-10;
    
    // creating the output variables
    plhs[0] = mxCreateDoubleMatrix(output_length, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(output_length, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(output_length, 1, mxREAL);
    plhs[3] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    output.costs = mxGetPr(plhs[0]);
    output.time = mxGetPr(plhs[1]);
    output.accuracy = mxGetPr(plhs[2]);
    output.grad_calc = (int*) mxGetData(plhs[3]); 
    
    // step 1.a. Transform the data to LibSVM format
    int err = 0;
    if(!mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2])) {
        mexPrintf("Error: label vector and instance matrix must be double\n");
        fake_answer(nlhs, plhs);
        return;
    }
    if(mxIsSparse(prhs[3]) || mxIsSparse(prhs[4])) {
        mexPrintf("Error: label vector should not be in sparse format");
        fake_answer(nlhs, plhs);
        return;
    }
    
	input.x = Malloc(struct feature_node*, options.l);
	input.y = Malloc(double, options.l);
    input.x_test = Malloc(struct feature_node*, options.l_test);
	input.y_test = Malloc(double, options.l_test);
    input.l = options.l;
    input.l_test = options.l_test;
    input.w = w;
    // for training data
    if(mxIsSparse(prhs[1])) {
        err = read_problem_sparse(prhs[1], prhs[3], input.x, input.y, options.l);
    } else {
        mexPrintf("Training_instance_matrix must be sparse; "
            "use sparse(Training_instance_matrix) first\n");
        clear_params_memory(&input);
        fake_answer(nlhs, plhs);
        return;
    }
    // for test data
    if(mxIsSparse(prhs[2])) {
        if (x_space!=NULL) {
            free(x_space);
        }
        err = read_problem_sparse(prhs[2], prhs[4], input.x_test, input.y_test, options.l_test);
    } else {
        mexPrintf("Testing_instance_matrix must be sparse; "
            "use sparse(Testing_instance_matrix) first\n");
        clear_params_memory(&input);
        fake_answer(nlhs, plhs);
        return;
    }
    options.p = input.p;    // bias term can increase it by one.
    
    // step 2. Call problem class and load the data and other options. 
    problem *prob = NULL;
    
    switch(problem_type) {
        case LOGISTIC_REGRESSION_L2R:            
            prob = new logistic_regression_l2r(&input, &options);
            break;
        case SVM_L2R_L2L:            
            prob = new svm_l2r_l2l(&input, &options);
            break;
        default:
            mexPrintf("Please carefully select one of the given problems.\n");
            exit(1);
    }
    
    // step 3. Call method class and load the options.
    method *solver = NULL;
    
    switch(options.method_type) {
        case STRON:
            solver = new stron(prob, &options);
            break;
        case STRON_SVRG:
            solver = new stron_svrg(prob, &options);
            break;
        case STRON_PCG_PROG:
            solver = new stron_pcg_prog(prob, &options);
            break;
        case STRON_PCG:
            solver = new stron_pcg(prob, &options);
            break;
        case TRON:
            solver = new tron(prob, &options);
            break;
        case SQN:
        case SVRG_SQN:
        case SVRG_LBFGS:    
            solver = new sqn(prob, &options);
            break;
        case LBFGS:
            solver = new lbfgs(prob, &options);
            break;
        case NEWTON_CG:
            solver = new newtoncg(prob, &options);
            break;
        default:
            mexPrintf("Please carefully select one of the given solvers.\n");
            exit(1);
    }
    
    // step 4. Excecute the method and pass the outputs to MATLAB.
    solver->solve(w, &output);
    
    // step 5. Clear the memory for before exiting.
    clear_params_memory(&input);
    if(x_space!=NULL) {
        free(x_space);
    }
    if(solver!=NULL) {
        delete solver; // base class destructor should be virtual.
    }
    if(prob!=NULL) {
        delete prob;
    }
}