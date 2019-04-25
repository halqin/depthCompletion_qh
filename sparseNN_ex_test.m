% MNIST_EXAMPLE
% Minimal demonstration of AutoNN training of a CNN on MNIST.
% The task is handwritten digit recognition.
%
% This example can be called with different name-value pairs, see the
% script below for a full list. Examples:
%
%  mnist_example                          % use defaults
%  mnist_example('learningRate', 0.0005)  % a lower learning rate
%  mnist_example('learningRate', 0.0005, 'gpu', [])  % no GPU
%  mnist_example('dataDir', '~/mnist')  % dataset path (downloaded automatically)
%  mnist_example('resultsDir', '~/out', 'savePlot', true)  % plot to ~/out
%

function [net, stats] = sparseNN_ex_test(varargin)
  % options (override by calling script with name-value pairs)
  opts.dataDir = [vl_rootnn() '/Users/Hall/convnn/data'] ;  % MNIST data location
  load('/Users/Hall/convnn/data/imdb_sparse_100.mat');
  opts.resultsDir = [vl_rootnn() 'data/mnist/cnn/mnist-example'] ;  % results location
  opts.numEpochs = 1 ;  % number of epochs
  opts.batchSize = 128 ;  % batch size
  opts.learningRate = 0.001 ;  % learning rate
  opts.solver = solvers.SGD() ;  % solver instance to use (type 'help solvers' for a list)
  opts.gpu = [] ;  % GPU index, empty for CPU mode
  opts.savePlot = false ;  % whether to save the plot as a PDF file
  
  opts = vl_argparse(opts, varargin) ;  % let user override options
  
  channels = 16;
    expansion = [1,2,4,4,4,8]; % the factors used to expand the channel number

    % depth U-net pathway %
    fsLow = [3 , 3]; padLow = floor(fsLow(1)/2);
    fsMed = [7 , 7]; padMed = floor(fsMed(1)/2);
    fsHigh= [11 , 11]; padHigh= floor(fsHigh(1)/2);
  
  
  try run('/Users/Hall/convnn/autonn-master/autonn-master/setup_autonn.m') ; catch; end  % add AutoNN to the path
  mkdir(opts.resultsDir) ;
  

  % build network. we could also have used:
  %   output = models.LeNet();
  % or any other model from 'autonn/matlab/+models/'.
  
  % create network inputs
  images = Input('gpu', true) ;
  labels = Input() ;
  
%   conv1 = vl_nnconv(images, 'size', [5, 5, 1, 20], 'weightScale', 0.01, 'pad', 2) ;
%   pool1 = vl_nnpool(conv1, 3, 'stride', 1, 'pad', 1) ;
  mask0 = single(images ~= 0);
  conv0_mul = images.*mask0;
  conv1 = vl_nnconv(conv0_mul, 'size', [fsHigh(1), fsHigh(2), 1, expansion(1)*channels], 'stride',1,'pad', 5 );
  norm = sum(sum(mask0));
  conv1_1 = conv1 ./norm;
  mask2 = vl_nnpool(mask0, 11, 'method', 'max', 'stride', 1 , 'pad' ,5);
  cat1 = vl_nnconcat({conv1_1, mask2},3, []);
  output = vl_nnconv(cat1, 'size', [1, 1, 17, 1], 'stride',1,'pad', 0 );

  
  
  
  
  % create losses
%   objective = vl_nnloss(output, labels, 'loss', 'softmaxlog') ;
  error = vl_nnloss(output, labels, 'loss', 'mae') ;

  % assign layer names automatically, and compile network
  Layer.workspaceNames() ;
  net = Net(error) ;

  % set solver learning rate
  solver = opts.solver ;
  solver.learningRate = opts.learningRate ;
  
  % initialize dataset
%   dataset = datasets.MNIST(opts.dataDir, 'batchSize', opts.batchSize) ;
  
  % compute average objective and error
  stats = Stats({'error'}) ;
  
  % enable GPU mode
  net.useGpu(opts.gpu) ;

  for epoch = 1:opts.numEpochs
    % training phase
    for batch = 1:1
      % draw samples
%       [images, labels] = dataset.get(batch) ;
      images = imdb.images.data(:,:,4,batch);
      labels = imdb.images.labels(:,:,:,batch);
%       montage(images); % plot the input images
      % evaluate network to compute gradients
      net.eval({'images', images, 'labels', labels}) ;
      
      % take one SGD step
      solver.step(net) ;
      
      % get current objective and error, and update their average
      stats.update(net) ;
      stats.print() ;
    end
    % push average objective and error (after one epoch) into the plot
    stats.push('train') ;

    % validation phase
    for batch = dataset.val()
      [images, labels] = dataset.get(batch) ;

      net.eval({'images', images, 'labels', labels}, 'test') ;

      stats.update(net) ;
      stats.print() ;
    end
    stats.push('val') ;

    % plot statistics
    stats.plot('figure', 1) ;
    if opts.savePlot && ~isempty(opts.resultsDir)
      print(1, [opts.resultsDir '/plot.pdf'], '-dpdf') ;
    end
  end
    
    % plot output of layers 
    figure(3) ; vl_tshow(net.getValue('conv3')); title('Conv3 filters') ;
%     figure(4) ; vl_tshow(net.getValue('conv2_filters')) ; title('Conv2 filters') ;
%     figure(5) ; vl_tshow(net.getValue('x_filters')) ; title('Conv3 filters') ;
  
  

  % save results
  if ~isempty(opts.resultsDir)
    save([opts.resultsDir '/results.mat'], 'net', 'stats', 'solver') ;
  end
end


function sparseconv()
    
end 

