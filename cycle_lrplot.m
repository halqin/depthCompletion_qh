
base_lr = 0.000001;
max_lr = 0.002;

epoch = 40;
batchsize = 10;
num_im  = 90;
iteration_epoch = num_im/batchsize; 
iteration_all = iteration_epoch*epoch;
stepsize = iteration_epoch/2;

for i = 1:iteration_all
    x(i) = cycle_lr(i, stepsize, base_lr, max_lr); 
end

plot(x)

function lr =  cycle_lr(iteration, stepsize, base_lr, max_lr)
    cycle = floor(1+ iteration/(2*stepsize));
     x = abs(iteration/stepsize - 2 * cycle + 1);
     lr = base_lr + (max_lr - base_lr) * max(0, (1-x)); 
end