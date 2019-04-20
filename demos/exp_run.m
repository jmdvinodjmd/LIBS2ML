% Experimental Demostration
% This file is used to demostrate the use of LIBS2ML library.
% This file is part of LIBS2ML.
% Created by V. K. Chauhan on Sept. 15, 2018
% Modified by V. K. Chauhan on Apr. 17, 2019
%% Clearing everyting from workspace
clc; clear all; close all;
%% Load dataset
load('data/news20');
X = [ones(size(X,1),1) X];
[l_total, p] = size(X);
X = X';y = y(:)';
%% split data into train and test data 
rand('seed', 2018); randn('seed', 2018);
perm_idx = randperm(l_total);
X = X(:,perm_idx); y = y(perm_idx);
l = floor(l_total * 0.8); % 80:20 for train:test
x_train = X(:,1:l); y_train = y(1:l)';
x_test = X(:,l+1:end); y_test = y(l+1:end)';
clear X; clear y; clear perm_idx;
%% common experimental setup
batch_size = floor(l*0.1); % Currently 10% batch size taken
max_iters = 15;
l_train = size(x_train, 2);
lambda =  1/l_train; %1/l_train 1e-2
cg_iters = 25;
%% Running Experiments
% Methods: TRON, STRON, STRON_PROG, STRON_PCG_PROG, SQN, SVRG_LBFGS, SVRG_SQN, LBFGS, NEWTON_CG
% problems: SVM_L2R_L2L LOGISTIC_REGRESSION_L2R
w{1} = zeros(p,1);
[info{1}.cost, info{1}.time, info{1}.accuracy, info{1}.grads] = interface(w{1}, x_train, x_test, y_train,...
      y_test, lambda, batch_size, max_iters, 'TRON', 'LOGISTIC_REGRESSION_L2R', cg_iters);
w{2} = zeros(p,1);
[info{2}.cost, info{2}.time, info{2}.accuracy, info{2}.grads] = interface(w{2}, x_train, x_test, y_train,...
      y_test, lambda, batch_size, max_iters, 'STRON', 'LOGISTIC_REGRESSION_L2R', cg_iters);
%% Calculate the optimal value to plot optimality gap
opt_obj = calc_optimal(info);
%% Display Results: x- time and epoch, y-accuracy, optimality_gap
fig1 = plot_graph(opt_obj, 'time','optimality_gap', {'TRON', 'STRON'}, info, [50 100 600 500]);
fig2 = plot_graph(opt_obj, 'time','accuracy', {'TRON', 'STRON'}, info, [700 100 600 500]);
% fig3 = plot_graph(opt_obj, 'epoch','optimality_gap', {'TRON', 'STRON'}, info, [50 100 600 500]);
% fig4 = plot_graph(opt_obj, 'epoch','accuracy', {'TRON', 'STRON'}, info, [700 100 600 500]);
%% The End