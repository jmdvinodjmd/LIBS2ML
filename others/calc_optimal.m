%% Calculate the optimal value to plot optimality gap.
% instead of calculating an optimal value using some other method, we use
% the lowest value amongst all the solvers and subtract a small value
function opt = calc_optimal(info)
    opt = Inf;
    for i = 1:length(info)
    %     grad_calc = info{i}.grads;
    %     if(length(info{i}.cost)> (max_iters + 1))
    %         info{i}.grads = 0:grad_calc:(length(info{i}.cost)-1)*grad_calc;
    %     else
    %         info{i}.grads = 0:grad_calc:max_iters*grad_calc;
    %     end
    %     fprintf('\nAlgorithm: %d\n', i);
        for j=1:length(info{i}.cost)
           if (opt >  info{i}.cost(j)) && (info{i}.cost(j) > 0)
               opt = info{i}.cost(j);           
           end
        end    
    end
    opt = opt - 1e-17;
    % fprintf('Opt: %.15f\n', opt);
end