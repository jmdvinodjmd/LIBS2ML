% Experimental Demostration
% This file is used to reproduce the results reported in STRON (Chauhan et al 2020) and presents
% comparative study of methods using SVM.
% This file is part of LIBS2ML.
% Created by V. K. Chauhan on Sept. 15, 2018
% Last Modified by V. K. Chauhan on August 08, 2021
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
dataset=('news20.binary.mat');
% dataset=('rcv1-train.binary.mat');
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
% maintain log of command line output
% diary(logs);
load(strcat(directory,dataset));
fprintf('Data loaded: %s\n', dataset);
X = [ones(size(X,1),1) X];
[l_total,p] = size(X);
X =X';y = y(:)';
%% 
% Normalization
% sum1 = 1./sqrt(sum(X.^2, 1));
% if abs(sum1(1) - 1) > 10^(-10)
%     X = X.*repmat(sum1, p, 1);
% end
% clear sum1;
rand('seed', 2018); randn('seed', 2018);
perm_idx = randperm(l_total);
X = X(:,perm_idx);
y = y(perm_idx);
% split data into train and test data 80:20
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
w_init = zeros(p,1);
batch_size = floor(l*0.1);
options.batch_size = batch_size;
max_iters = 15;
l_train = size(x_train,2);
lambda =  1/l_train; %1/l_train 1e-2
cg_iters = 25;
%% Methods: TRON, STRON, STRON_SVRG, STRON_PCG, STRON_PCG_PROG, SQN, SVRG_LBFGS, SVRG_SQN, LBFGS, NEWTON_CG
% problems: SVM_L2R_L2L LOGISTIC_REGRESSION_L2R
w{1} = zeros(p,1);
[info{1}.cost, info{1}.time, info{1}.accuracy, info{1}.grad_calc_count] = interface(w{1}, x_train, x_test, y_train,...
      y_test, lambda, batch_size, max_iters, 'SVRG_LBFGS', 'SVM_L2R_L2L');

w{2} = zeros(p,1);
[info{2}.cost, info{2}.time, info{2}.accuracy, info{2}.grad_calc_count] = interface(w{2}, x_train, x_test, y_train,...
      y_test, lambda, batch_size, max_iters, 'STRON', 'SVM_L2R_L2L', cg_iters);
    
w{3} = zeros(p,1);
[info{3}.cost, info{3}.time, info{3}.accuracy, info{3}.grad_calc_count] = interface(w{3}, x_train, x_test, y_train,...
      y_test, lambda, batch_size, max_iters, 'SVRG_SQN', 'SVM_L2R_L2L');
  
w{4} = zeros(p,1);
[info{4}.cost, info{4}.time, info{4}.accuracy, info{4}.grad_calc_count] = interface(w{4}, x_train, x_test, y_train,...
      y_test, lambda, batch_size, max_iters, 'NEWTON_CG', 'SVM_L2R_L2L', cg_iters);
  
w{5} = zeros(p,1);
[info{5}.cost, info{5}.time, info{5}.accuracy, info{5}.grad_calc_count] = interface(w{5}, x_train, x_test, y_train,...
      y_test, lambda, batch_size, max_iters, 'TRON', 'SVM_L2R_L2L', cg_iters);
%% Calculate the optimal value to plot optimality gap
opt_obj = calc_optimal(info);
%% Display Results
% 5
fig1 = plot_graph(opt_obj, 'time','optimality_gap', {'SVRG\_LBFGS', 'STRON', 'SVRG\_SQN', 'NewtonCG', 'TRON'}, info, [50 100 600 500]);
% saveas(fig1,'result1.eps', 'epsc');
fig2 = plot_graph(opt_obj, 'time','accuracy', {'SVRG\_LBFGS', 'STRON', 'SVRG\_SQN', 'NewtonCG', 'TRON'}, info, [700 100 600 500]);
% saveas(fig2,'result2.eps', 'epsc');
% fig3 = display_graphh(opt_obj, 'time','cost', {'STRON', 'TRON', 'SVRG\_LBFGS', 'SVRG\_SQN', 'SQN'}, w, info, [50 100 600 500]);
% saveas(fig3,'result3.eps', 'epsc');
% fig4 = display_graphh(opt_obj, 'time','accuracy', {'STRON', 'TRON', 'SVRG\_LBFGS', 'SVRG\_SQN', 'SQN'}, w, info, [700 100 600 500]);
% saveas(fig4,'result4.eps', 'epsc');

% 4
% fig3 = display_graphh(opt_obj, 'time','cost', {'LBFGS', 'SVRG\_LBFGS',  'SVRG\_SQN', 'TRON'}, w, info, [50 100 600 500]);
% saveas(fig3,'result3.eps', 'epsc');
% fig4 = display_graphh(opt_obj, 'time','accuracy', {'LBFGS', 'SVRG\_LBFGS', 'SVRG\_SQN', 'TRON'}, w, info, [700 100 600 500]);
% saveas(fig4,'result4.eps', 'epsc');

% fig3 = display_graphh(opt_obj, 'time','cost', {'STRON', 'TRON', 'SVRG\_LBFGS', 'SVRG\_SQN'}, w, info, [50 100 600 500]);
% saveas(fig3,'result3.eps', 'epsc');
% fig4 = display_graphh(opt_obj, 'time','accuracy', {'STRON', 'TRON', 'SVRG\_LBFGS', 'SVRG\_SQN'}, w, info, [700 100 600 500]);
% saveas(fig4,'result4.eps', 'epsc');
% 


% 3
% fig3 = display_graphh(opt_obj, 'time','optimality_gap', {'TRON', 'STRON', 'STRON1'}, w, info, [50 100 600 500]);
% saveas(fig3,'result3.eps', 'epsc');
% fig4 = display_graphh(opt_obj, 'time','accuracy', {'TRON', 'STRON', 'STRON1'}, w, info, [700 100 600 500]);
% saveas(fig4,'result4.eps', 'epsc');


% % 2
% fig3 = display_graphh(opt_obj, 'time','optimality_gap', {'NewtonCG', 'STRON'}, w, info, [50 100 600 500]);
% saveas(fig3,'result3.eps', 'epsc');
% fig4 = display_graphh(opt_obj, 'time','accuracy', {'NewtonCG', 'STRON'}, w, info, [700 100 600 500]);
% saveas(fig4,'result4.eps', 'epsc');


% % fig3 = display_graphh(opt_obj, 'time','optimality_gap', {'LBFGS', 'SVRG\_LBFGS'}, w, info, [50 100 600 500]);
% % saveas(fig3,'result3.eps', 'epsc');
% % fig4 = display_graphh(opt_obj, 'time','accuracy', {'LBFGS', 'SVRG\_LBFGS' }, w, info, [700 100 600 500]);
% % saveas(fig4,'result4.eps', 'epsc');
% 
% % 
% % % displaying the summary of results:
% % % fprintf('Methods  : MBN, \t\t MBN-SVRG \t\t MBN-VRSGD\n');
% % % fprintf('Accuracy : %.4f \t\t %.4f \t\t %.4f\n', info{1}.accuracy(end), info{2}.accuracy(end), info{3}.accuracy(end));
% % % fprintf('Time     : %.4f \t\t %.4f \t\t %.4f\n', info{1}.time(end), info{2}.time(end), info{3}.time(end)); 
% % % fprintf('Cost     : %.12f \t %.12f \t %.12f\n', info{1}.cost(end), info{2}.cost(end), info{3}.cost(end)); 
% diary(logs);
