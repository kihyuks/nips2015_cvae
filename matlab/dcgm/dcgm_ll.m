function [logpy, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
    dcgm_ll(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
    do_recurrent, alpha, recmodel, genmodel, predmodel, nsample)

if ~exist('do_recurrent', 'var'),
    do_recurrent = 0;
end
if ~exist('alpha', 'var'),
    alpha = 1;
end
if ~exist('recmodel', 'var'),
    recmodel = 'qzxy';
end
if ~exist('genmodel', 'var'),
    genmodel = 'pzx';
end
if ~exist('predmodel', 'var'),
    predmodel = 'pyxz';
end
if ~exist('gradcheck', 'var'),
    gradcheck = 0;
end

dropout = 0;
doder = 1;
disample = 0;

if do_recurrent,
    % recurrent encoding
    if alpha > 0,
        % hybrid, CVAE, GSNN
        [logpy, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
            cvae_recurrent_cll(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
            alpha, recmodel, genmodel, predmodel, gradcheck, dropout, disample, nsample);
    else
        % CNN, GSNN
        [logpy, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
            cnn_recurrent_cll(x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
            genmodel, predmodel, gradcheck, dropout, disample, nsample);
    end
else
    if alpha > 0,
        % hybrid, CVAE
        [logpy, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
            cvae_cll(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
            alpha, recmodel, predmodel, gradcheck, dropout, disample);
    else
        % CNN, GSNN
        [logpy, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
            cnn_cll(x, net_x2y, net_cnn, net_cls, res_z2y, res_x2z, ...
            res_x2y, res_cnn, res_cls, gradcheck, dropout, disample);
    end
end

return;



% -----------------------------------------------------------------------
%   conditional log-likelihood of CNN or GSNN
%   (i.e., CNN + stochastic neuron)
%
%   supports: multi-scale, flat encoding
%
%   x   : input data
%   target data should have been fed to net_cls.layers{end}.data
% -----------------------------------------------------------------------

function [logpy, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
    cnn_cll(x, net_x2y, net_cnn, net_cls, res_z2y, res_x2z, res_x2y, ...
    res_cnn, res_cls, gradcheck, dropout, disample)


[logpy, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = cnn_cll_sub...
    (x, net_x2y, net_cnn, net_cls, res_z2y, res_x2z, res_x2y, ...
    res_cnn, res_cls, gradcheck, dropout, disample);

logpy = logpy(:);

return;


function [logpy, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = cnn_cll_sub...
    (x, net_x2y, net_cnn, net_cls, res_z2y, res_x2z, res_x2y, ...
    res_cnn, res_cls, gradcheck, dropout, disample)

if ~exist('gradcheck', 'var'),
    gradcheck = 0;
end
if ~exist('dropout', 'var'),
    dropout = 0;
end
if ~exist('disample', 'var'),
    disample = 0;
end

opts = struct(...
    'gradcheck', gradcheck, ...
    'disableDropout', dropout, ...
    'disableSampling', disample);

gpuMode = isa(x, 'gpuArray');
if gradcheck,
    one = double(1);
else
    one = single(1);
end

if gpuMode,
    one = gpuArray(one);
end

nscale = length(net_cls);


% -------------------------------------------------------------------------
%                                                     generation (x -> y)
% -------------------------------------------------------------------------

res_x2y{1} = mcn_ff(net_x2y{1}, x, [], res_x2y{1}, opts);
for j = 2:nscale,
    res_x2y{j} = mcn_ff(net_x2y{j}, res_x2y{j-1}(end).x, [], res_x2y{j}, opts);
end
res_cnn{nscale} = mcn_ff(net_cnn{nscale}, res_x2y{nscale}(end).x, [], res_cnn{nscale}, opts);


% -------------------------------------------------------------------------
%                                             generation (x,y -> z -> hy)
% -------------------------------------------------------------------------

res_cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x, [], res_cls{nscale}, opts);

% logpz, logqz
switch strtok(net_cls{end}.layers{1}.type, '_'),
    case 'bernoulli',
        logpy = gather(compute_bernoulli_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
    case 'gaussian',
        logpy = gather(compute_gaussian_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
    case 'softmax',
        logpy = gather(compute_softmax_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
end


return;



% -----------------------------------------------------------------------
%   conditional log-likelihood of CVAE or GSNN
%
%   supports: multi-scale, flat encoding
%
%   x   : input data
%   y   : output data (original scale)
%   multi-scale output should have been fed to net_cls.layers{end}.data
% -----------------------------------------------------------------------

function [logpy, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
    cvae_cll(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
    alpha, recmodel, predmodel, gradcheck, dropout, disample)

if ~exist('alpha', 'var'),
    alpha = 0;
end
if ~exist('recmodel', 'var'),
    recmodel = 'qzxy';
end
if ~exist('predmodel', 'var'),
    predmodel = 'pyxz';
end

[logpy, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
    cvae_cll_sub(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
    alpha, recmodel, predmodel, gradcheck, dropout, disample);

return;

function [logpy, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
    cvae_cll_sub(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
    alpha, recmodel, predmodel, gradcheck, dropout, disample)

if ~exist('alpha', 'var'),
    alpha = 1;
end
if ~exist('gradcheck', 'var'),
    gradcheck = 0;
end
if ~exist('dropout', 'var'),
    dropout = 0;
end
if ~exist('disample', 'var'),
    disample = 0;
end

opts = struct(...
    'gradcheck', gradcheck, ...
    'disableDropout', dropout, ...
    'disableSampling', disample);

gpuMode = isa(x, 'gpuArray');
if gradcheck,
    one = double(1);
else
    one = single(1);
end

if gpuMode,
    one = gpuArray(one);
end

nscale = length(net_cls);

if gpuMode,
    alpha = gpuArray(alpha);
end


% -----------------------------------------------------------------------
%                                                generation (x -> y)
% -----------------------------------------------------------------------

res_x2y{1} = mcn_ff(net_x2y{1}, x, [], res_x2y{1}, opts);
for j = 2:nscale,
    res_x2y{j} = mcn_ff(net_x2y{j}, res_x2y{j-1}(end).x, [], res_x2y{j}, opts);
end
res_cnn{nscale} = mcn_ff(net_cnn{nscale}, res_x2y{nscale}(end).x, [], res_cnn{nscale}, opts);


% -----------------------------------------------------------------------
%                                recognition & generation (x,y -> z)
% -----------------------------------------------------------------------

if strcmp(recmodel, 'qzxy'),
    res_xy2z = mcn_ff(net_xy2z, cat(3, x, y), [], res_xy2z, opts);
elseif strcmp(recmodel, 'qzy'),
    res_xy2z = mcn_ff(net_xy2z, y, [], res_xy2z, opts);
end
res_x2z = mcn_ff(net_x2z, x, [], res_x2z, opts);


% -----------------------------------------------------------------------
%                                                generation (z -> y)
% -----------------------------------------------------------------------

batchsize = size(x, 4);
logpy = zeros(nsample, batchsize);
logqz = zeros(nsample, batchsize);
logpz = zeros(nsample, batchsize);

for i = 1:nsample,
    if alpha == 0,
        % prediction
        res_z2y = mcn_ff(net_z2y, res_x2z(end).x, res_x2z(end).s, res_z2y, opts);
        if strcmp(predmodel, 'pyxz'),
            res_cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x + res_z2y(end).x, [], res_cls{nscale}, opts);
        elseif strcmp(predmodel, 'pyz'),
            res_cls{nscale} = mcn_ff(net_cls{nscale}, res_z2y(end).x, [], res_cls{nscale}, opts);
        end
        
        % logpy
        switch strtok(net_cls{end}.layers{1}.type, '_'),
            case 'bernoulli',
                logpy(i, :) = gather(compute_bernoulli_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
            case 'gaussian',
                logpy(i, :) = gather(compute_gaussian_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
            case 'softmax',
                logpy(i, :) = gather(compute_softmax_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
        end
    else
        % generation
        res_z2y = mcn_ff(net_z2y, res_xy2z(end).x, res_xy2z(end).s, res_z2y, opts);
        if strcmp(predmodel, 'pyxz'),
            res_cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x + res_z2y(end).x, [], res_cls{nscale}, opts);
        elseif strcmp(predmodel, 'pyz'),
            res_cls{nscale} = mcn_ff(net_cls{nscale}, res_z2y(end).x, [], res_cls{nscale}, opts);
        end
        
        % logpz, logqz
        logqz(i, :) = gather(compute_gaussian_loss(res_z2y(2).x, res_xy2z(end).x, res_xy2z(end).s, 1));
        logpz(i, :) = gather(compute_gaussian_loss(res_z2y(2).x, res_x2z(end).x, res_x2z(end).s, 1));
        
        % logpy
        switch strtok(net_cls{end}.layers{1}.type, '_'),
            case 'bernoulli',
                logpy(i, :) = gather(compute_bernoulli_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
            case 'gaussian',
                logpy(i, :) = gather(compute_gaussian_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
            case 'softmax',
                logpy(i, :) = gather(compute_softmax_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
        end
    end
end

logpy = logpy + logpz - logqz;

logpy = max(logpy, [], 1) + log(sum(exp(bsxfun(@minus, logpy, max(logpy, [], 1))), 1));
logpy = logpy - log(nsample);
logpy = logpy(:);

return;



% -----------------------------------------------------------------------
%   conditional log-likelihood of CNN or GSNN
%   (i.e., CNN + stochastic neuron)
%
%   supports: multi-scale, recurrent encoding
%
%   x   : input data
%   target data is should have been fed to nec_cls.layers{end}.data
% -----------------------------------------------------------------------

function [logpy, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
    cnn_recurrent_cll(x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
    genmodel, predmodel, gradcheck, dropout, disample, nsample)

if ~exist('genmodel', 'var'),
    genmodel = 'pzx';
end
if ~exist('predmodel', 'var'),
    predmodel = 'pyxz';
end

[logpy, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = cnn_recurrent_cll_sub...
    (x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
    genmodel, predmodel, gradcheck, dropout, disample, nsample);

return;


function [logpy, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = cnn_recurrent_cll_sub...
    (x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
    genmodel, predmodel, gradcheck, dropout, disample, nsample)

if ~exist('gradcheck', 'var'),
    gradcheck = 0;
end
if ~exist('dropout', 'var'),
    dropout = 0;
end
if ~exist('disample', 'var'),
    disample = 0;
end

opts = struct(...
    'gradcheck', gradcheck, ...
    'disableDropout', dropout, ...
    'disableSampling', disample);

gpuMode = isa(x, 'gpuArray');
if gradcheck,
    one = double(1);
else
    one = single(1);
end

if gpuMode,
    one = gpuArray(one);
end

nscale = length(net_cls);


% -----------------------------------------------------------------------
%                                                     generation (x -> y)
% -----------------------------------------------------------------------

res_x2y{1} = mcn_ff(net_x2y{1}, x, [], res_x2y{1}, opts);
for j = 2:nscale,
    res_x2y{j} = mcn_ff(net_x2y{j}, res_x2y{j-1}(end).x, [], res_x2y{j}, opts);
end
res_cnn{nscale} = mcn_ff(net_cnn{nscale}, res_x2y{nscale}(end).x, [], res_cnn{nscale}, opts);
if strcmp(predmodel, 'pyxz'),
    scale_factor = 2;
elseif strcmp(predmodel, 'pyz'),
    scale_factor = 1;
end
res_cls{nscale} = mcn_ff(net_cls{nscale}, scale_factor*res_cnn{nscale}(end).x, [], res_cls{nscale}, opts);


% -----------------------------------------------------------------------
%                                       generation (x,y -> z -> hy)
% -----------------------------------------------------------------------

if strcmp(genmodel, 'pzx'),
    res_x2z = mcn_ff(net_x2z, cat(3, x, res_cls{nscale}(1).x), [], res_x2z, opts);
elseif strcmp(genmodel, 'pzy'),
    res_x2z = mcn_ff(net_x2z, res_cls{nscale}(1).x, [], res_x2z, opts);
end

batchsize = size(x, 4);
logpy = zeros(nsample, batchsize);

for i = 1:nsample,
    % prediction
    res_z2y = mcn_ff(net_z2y, res_x2z(end).x, res_x2z(end).s, res_z2y, opts);
    if strcmp(predmodel, 'pyxz'),
        res_cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x + res_z2y(end).x, [], res_cls{nscale}, opts);
    elseif strcmp(predmodel, 'pyz'),
        res_cls{nscale} = mcn_ff(net_cls{nscale}, res_z2y(end).x, [], res_cls{nscale}, opts);
    end
    
    % logpy
    switch strtok(net_cls{end}.layers{1}.type, '_'),
        case 'bernoulli',
            logpy(i, :) = gather(compute_bernoulli_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
        case 'gaussian',
            logpy(i, :) = gather(compute_gaussian_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
        case 'softmax',
            logpy(i, :) = gather(compute_softmax_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
    end
end

if nsample > 1,
    logpy = max(logpy, [], 1) + log(sum(exp(bsxfun(@minus, logpy, max(logpy, [], 1))), 1));
    logpy = logpy - log(nsample);
end
logpy = logpy(:);

return;



% -----------------------------------------------------------------------
%   conditional log-likelihood of CVAE or GSNN
%
%   supports: multi-scale, recurrent encoding
%
%   x   : input data
%   y   : output data (original scale)
%   multi-scale output should have been fed to net_cls.layers{end}.data
% -----------------------------------------------------------------------

function [logpy, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
    cvae_recurrent_cll(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
    alpha, recmodel, genmodel, predmodel, gradcheck, dropout, disample, nsample)

if ~exist('alpha', 'var'),
    alpha = 0;
end
if ~exist('recmodel', 'var'),
    recmodel = 'qzxy';
end
if ~exist('genmodel', 'var'),
    genmodel = 'pzx';
end
if ~exist('predmodel', 'var'),
    predmodel = 'pyxz';
end

[logpy, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
    cvae_recurrent_cll_sub(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
    alpha, recmodel, genmodel, predmodel, gradcheck, dropout, disample, nsample);

return;

function [logpy, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
    cvae_recurrent_cll_sub(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
    alpha, recmodel, genmodel, predmodel, gradcheck, dropout, disample, nsample)

if ~exist('alpha', 'var'),
    alpha = 1;
end
if ~exist('gradcheck', 'var'),
    gradcheck = 0;
end
if ~exist('dropout', 'var'),
    dropout = 0;
end
if ~exist('disample', 'var'),
    disample = 0;
end

opts = struct(...
    'gradcheck', gradcheck, ...
    'disableDropout', dropout, ...
    'disableSampling', disample);

gpuMode = isa(x, 'gpuArray');
if gradcheck,
    one = double(1);
else
    one = single(1);
end

if gpuMode,
    one = gpuArray(one);
end

nscale = length(net_cls);

if gpuMode,
    alpha = gpuArray(alpha);
end


% -----------------------------------------------------------------------
%                                                generation (x -> y)
% -----------------------------------------------------------------------

res_x2y{1} = mcn_ff(net_x2y{1}, x, [], res_x2y{1}, opts);
for j = 2:nscale,
    res_x2y{j} = mcn_ff(net_x2y{j}, res_x2y{j-1}(end).x, [], res_x2y{j}, opts);
end
res_cnn{nscale} = mcn_ff(net_cnn{nscale}, res_x2y{nscale}(end).x, [], res_cnn{nscale}, opts);
if strcmp(predmodel, 'pyxz'),
    scale_factor = 2;
elseif strcmp(predmodel, 'pyz'),
    scale_factor = 1;
end
res_cls{nscale} = mcn_ff(net_cls{nscale}, scale_factor*res_cnn{nscale}(end).x, [], res_cls{nscale}, opts);


% -----------------------------------------------------------------------
%                                recognition & generation (x,y -> z)
% -----------------------------------------------------------------------

if strcmp(recmodel, 'qzxy'),
    res_xy2z = mcn_ff(net_xy2z, cat(3, x, y), [], res_xy2z, opts);
elseif strcmp(recmodel, 'qzy'),
    res_xy2z = mcn_ff(net_xy2z, y, [], res_xy2z, opts);
end
if strcmp(genmodel, 'pzx'),
    res_x2z = mcn_ff(net_x2z, cat(3, x, res_cls{nscale}(1).x), [], res_x2z, opts);
elseif strcmp(genmodel, 'pzy'),
    res_x2z = mcn_ff(net_x2z, res_cls{nscale}(1).x, [], res_x2z, opts);
end


% -----------------------------------------------------------------------
%                                                generation (z -> y)
% -----------------------------------------------------------------------

batchsize = size(x, 4);
logpy = zeros(nsample, batchsize);
logqz = zeros(nsample, batchsize);
logpz = zeros(nsample, batchsize);

for i = 1:nsample,
    if alpha == 0,
        % prediction
        res_z2y = mcn_ff(net_z2y, res_x2z(end).x, res_x2z(end).s, res_z2y, opts);
        if strcmp(predmodel, 'pyxz'),
            res_cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x + res_z2y(end).x, [], res_cls{nscale}, opts);
        elseif strcmp(predmodel, 'pyz'),
            res_cls{nscale} = mcn_ff(net_cls{nscale}, res_z2y(end).x, [], res_cls{nscale}, opts);
        end
        
        % logpy
        switch strtok(net_cls{end}.layers{1}.type, '_'),
            case 'bernoulli',
                logpy(i, :) = gather(compute_bernoulli_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
            case 'gaussian',
                logpy(i, :) = gather(compute_gaussian_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
            case 'softmax',
                logpy(i, :) = gather(compute_softmax_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
        end
    else
        % generation
        res_z2y = mcn_ff(net_z2y, res_xy2z(end).x, res_xy2z(end).s, res_z2y, opts);
        if strcmp(predmodel, 'pyxz'),
            res_cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x + res_z2y(end).x, [], res_cls{nscale}, opts);
        elseif strcmp(predmodel, 'pyz'),
            res_cls{nscale} = mcn_ff(net_cls{nscale}, res_z2y(end).x, [], res_cls{nscale}, opts);
        end
        
        % logpz, logqz
        logqz(i, :) = gather(compute_gaussian_loss(res_z2y(2).x, res_xy2z(end).x, res_xy2z(end).s, 1));
        logpz(i, :) = gather(compute_gaussian_loss(res_z2y(2).x, res_x2z(end).x, res_x2z(end).s, 1));
        
        % logpy
        switch strtok(net_cls{end}.layers{1}.type, '_'),
            case 'bernoulli',
                logpy(i, :) = gather(compute_bernoulli_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
            case 'gaussian',
                logpy(i, :) = gather(compute_gaussian_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
            case 'softmax',
                logpy(i, :) = gather(compute_softmax_loss(net_cls{nscale}.layers{end}.data, res_cls{nscale}(end-1).x, res_cls{nscale}(end-1).s, 1));
        end
    end
end

logpy = logpy + logpz - logqz;

logpy = max(logpy, [], 1) + log(sum(exp(bsxfun(@minus, logpy, max(logpy, [], 1))), 1));
logpy = logpy - log(nsample);
logpy = logpy(:);

return