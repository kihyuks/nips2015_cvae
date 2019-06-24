% -----------------------------------------------------------------------
%   numvx       : size(xtr, 3)
%   numvy       : size(ytr, 3)
%   numhid      : number of hidden units for each layer
%   l2reg       : weight decay parameter
%   dropout     :
%   nsample     : number of samples for VAE (default = 1)
%   std_init    : weight initialization std
%   alpha       :
%   numft       : number of filters
%   numpool     : pooling size
%   stride      : convolution stride
%   stotype     : stochastic type
%                 0 - deterministic
%                 1 - gaussian sampling with KL divergence loss
%                 2 - gaussian sampling without loss
%   convtype    : 'same' convolution for most cases
%   pltype      : pooling type
%   rectype     : nonlinearity type
%
% -----------------------------------------------------------------------

function [params_xy2y, params_x2z, params_x2y] = mk_params(dataset, optgpu, savedir, recmodel, genmodel, ...
    do_recurrent, typein, typeout, numvx, numvy, l2reg, dropout, nsample, std_init, alpha, ...
    numhid, numft, numpool, stride, stotype, convtype, pltype, rectype, ...
    numhid_c, numft_c, numpool_c, stride_c, stotype_c, convtype_c, pltype_c, rectype_c)

if ~exist('do_recurrent', 'var'),
    do_recurrent = 0;
end
if ~exist('recmodel', 'var'),
    recmodel = 'qzxy';
end
if ~exist('genmodel', 'var'),
    genmodel = 'pzx';
end


% parameter for x, y -> y (q(z|x,y) -> p(y|x,z))
switch recmodel,
    case 'qzxy',
        numin = numvx+numvy;
    case 'qzy',
        numin = numvy;
end
params_xy2y = mk_params_sub(dataset, optgpu, savedir, typein, typeout, ...
    numin, numhid, numvy, numft, numpool, stride, stotype, ...
    convtype, pltype, rectype, l2reg, dropout, nsample, std_init, alpha);

% parameter for x -> z (p(z|x))
k = find(stotype == 3);
if isempty(k),
    k = find(stotype == 1);
    stotype_x2z = zeros(1, k);
    stotype_x2z(k) = 1;
else
    stotype_x2z = zeros(1, k);
    stotype_x2z(k) = 2;
end
if do_recurrent,
    switch genmodel,
        case 'pzx',
            numin = numvx+numvy;
        case 'pzy',
            numin = numvy;
    end
else
    numin = numvx;
end
params_x2z = mk_params_sub(dataset, optgpu, savedir, typein, typeout, ...
    numin, numhid(1:k), numvy, numft(1:k+1,:), numpool(1:k), stride(1:k+1), stotype_x2z, ...
    convtype, pltype, rectype, l2reg, dropout, nsample, std_init, alpha);

% parameter for x -> y (p(y|x))
params_x2y = mk_params_sub(dataset, optgpu, savedir, typein, typeout, ...
    numvx, numhid_c, numvy, numft_c, numpool_c, stride_c, stotype_c, ...
    convtype_c, pltype_c, rectype_c, l2reg, dropout, nsample, std_init, alpha);

return;


% -----------------------------------------------------------------------
%   make parameters (sub function)
% -----------------------------------------------------------------------

function params = mk_params_sub(dataset, optgpu, savedir, typein, typeout, ...
    numvis, numhid, numlab, numft, numpool, stride, stotype, ...
    convtype, pltype, rectype, l2reg, dropout, nsample, std_init, alpha)

params = struct(...
    'dataset', dataset, ...
    'prefix', dataset, ...
    'optgpu', optgpu, ...
    'savedir', savedir, ...
    'typein', typein, ...
    'typeout', typeout, ...
    'numvis', numvis, ...
    'numhid', numhid, ...
    'numlab', numlab, ...
    'numft', numft, ...
    'numpool', numpool, ...
    'stride', stride, ...
    'stotype', stotype, ...
    'rectype', rectype, ...
    'l2reg', l2reg, ...
    'dropout', dropout, ...
    'nsample', nsample, ...
    'std_init', std_init);

params.convtype = convtype;
params.pltype = pltype;
if exist('alpha', 'var'),
    params.alpha = alpha;
end

return;