% Experimental Demostration
% This file is used to reproduce the results reported in STRON (Chauhan et al 2018).
% This file is part of LIBS2ML.
% Created by V. K. Chauhan on Sept. 15, 2018
% Modified by V. K. Chauhan on Apr. 18, 2019
%% Clearing everyting from workspace
clc; clear all; close all;
%% Load data
directory = '/Volumes/My Data/PhD/Code/Data Sets/Original/matlab-formats/';
% dataset=('covtype.binary.mat');
% dataset=('gisette.mat');
% dataset=('ijcnn1.mat');
% dataset=('SUSY.mat');
% dataset=('HIGGS.mat');
% dataset=('epsilon.mat');
% dataset=('webspam-uni.mat');
% dataset=('news20.binary.mat');
dataset=('rcv1-train.binary.mat');
% dataset=('real-sim.mat');
% dataset=('avazu-site.mat');
% dataset=('avazu-app.mat'); %light
% dataset=('mnist.mat'); % multi-class
% dataset=('combined.mat'); % multi-class
% dataset=('SVHN.mat'); % multi-class
% dataset=('protein.mat'); % multi-class
% dataset=('madelon.mat');
% dataset= ('Adult');
% dataset=('heart_scale');
% dataset=('w8a');
% dataset=('mushroom');
fprintf('Loading Data...: %s\n', dataset);
% logs=strrep(strcat('logs/',dataset,'_',num2str(datestr(datenum(now),'yyyy-mm-dd HH:MM:SS'))),'.','_');
% diary(logs);
load(strcat(directory,dataset));
fprintf('Data loaded: %s\n', dataset);
X = [ones(size(X,1),1) X];
[l_total,p] = size(X);
X =X';y = y(:)';
%% Normalization
% sum1 = 1./sqrt(sum(X.^2, 1));
% if abs(sum1(1) - 1) > 10^(-10)
%     X = X.*repmat(sum1, p, 1);
% end
% clear sum1;
rand('seed', 2018); randn('seed', 2018);
perm_idx = randperm(l_total);
X = X(:,perm_idx);
y = y(perm_idx);
% split data into train and test data
% train data
l = floor(l_total * 0.8);
x_train = X(:,1:l);
y_train = y(1:l)';    
% test data
x_test = X(:,l+1:end);
y_test = y(l+1:end)';
% clearing memory
clear X; clear y; clear perm_idx;
%% common experimental setup
w_init = zeros(p,1); %randn(p,1), zeros(p,1)
batch_size = floor(l*0.1); %min(5000, floor(l*0.05));
options.batch_size = batch_size;
max_iters = 15;
l_train = size(x_train,2);
lambda =  1/l_train; %1/l_train 1e-2
cg_iters = 25;
%% Methods: TRON, STRON, STRON_SVRG, STRON_PCG, STRON_PCG_PROG, SQN, SVRG_LBFGS, SVRG_SQN, LBFGS, NEWTON_CG
% problems: SVM_L2R_L2L LOGISTIC_REGRESSION_L2R
w{1} = zeros(p,1);
[info{1}.cost, info{1}.time, info{1}.accuracy, info{1}.grad_calc_count] = interface(w{1}, x_train, x_test, y_train,...
      y_test, lambda, batch_size, max_iters, 'STRON_PCG_PROG', 'LOGISTIC_REGRESSION_L2R', cg_iters);
w{2} = zeros(p,1);
[info{2}.cost, info{2}.time, info{2}.accuracy, info{2}.grad_calc_count] = interface(w{2}, x_train, x_test, y_train,...
      y_test, lambda, batch_size, max_iters, 'STRON', 'LOGISTIC_REGRESSION_L2R', cg_iters);
w{3} = zeros(p,1);
[info{3}.cost, info{3}.time, info{3}.accuracy, info{3}.grad_calc_count] = interface(w{3}, x_train, x_test, y_train,...
      y_test, lambda, batch_size, max_iters, 'TRON', 'LOGISTIC_REGRESSION_L2R', cg_iters);
%% Calculate the optimal value to plot optimality gap
opt_obj = calc_optimal(info);
%% Display Results
% 3
fig1 = plot_graph(opt_obj, 'epoch','optimality_gap', {'STRON-PCG', 'STRON', 'TRON'}, info, [50 100 600 500]);
% saveas(fig1,'result1.eps', 'epsc');
fig2 = plot_graph(opt_obj, 'epoch','accuracy', {'STRON-PCG', 'STRON', 'TRON'}, info, [700 100 600 500]);
% saveas(fig2,'result2.eps', 'epsc');

% % % displaying the summary of results:
% % % fprintf('Methods  : MBN, \t\t MBN-SVRG \t\t MBN-VRSGD\n');
% % % fprintf('Accuracy : %.4f \t\t %.4f \t\t %.4f\n', info{1}.accuracy(end), info{2}.accuracy(end), info{3}.accuracy(end));
% % % fprintf('Time     : %.4f \t\t %.4f \t\t %.4f\n', info{1}.time(end), info{2}.time(end), info{3}.time(end)); 
% % % fprintf('Cost     : %.12f \t %.12f \t %.12f\n', info{1}.cost(end), info{2}.cost(end), info{3}.cost(end)); 
% diary(logs);