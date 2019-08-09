
base_lr = 0.000001;
max_lr = 0.002;

epoch = 40;
batchsize = 3;
num_im  = 461;
%iteration_epoch = num_im/batchsize; 
%iteration_all = iteration_epoch*epoch;
iteration_count = 0; 

for j = 1:epoch
    iteration_epoch = num_im/batchsize; 
    stepsize = iteration_epoch/4;
    x = batchprocess(batchsize, num_im, base_lr, max_lr,stepsize, iteration_count);
end 

% for i = 1:iteration_all
%     x(i) = cycle_lr(i, stepsize, base_lr, max_lr); 
% end


plot(x);

function lr =  cycle_lr(iteration, stepsize, base_lr, max_lr)
    cycle = floor(1+ iteration/(2*stepsize));
     x = abs(iteration/stepsize - 2 * cycle + 1);
     lr = base_lr + (max_lr - base_lr) * max(0, (1-x)); 
end



function x = batchprocess(batchsize, num_im, base_lr, max_lr, stepsize, iteration_count)
    for i = 1:batchsize:num_im
        iteration_count = iteration_count + 1; 
            x(iteration_count)= cycle_lr(iteration_count, stepsize, base_lr, max_lr); 
    end
end 




