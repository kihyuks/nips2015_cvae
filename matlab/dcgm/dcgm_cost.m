function [cost, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
    dcgm_cost(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
    do_recurrent, alpha, recmodel, genmodel, predmodel, gradcheck)

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
        % hybrid, CVAE
        [cost, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
            cvae_recurrent_cost(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
            alpha, recmodel, genmodel, predmodel, gradcheck, dropout, doder, disample);
    else
        % CNN, GSNN
        [cost, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
            cnn_recurrent_cost(x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
            genmodel, predmodel, gradcheck, dropout, doder, disample);
    end
else
    if alpha > 0,
        % hybrid, CVAE
        [cost, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
            cvae_cost(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
            alpha, recmodel, predmodel, gradcheck, dropout, doder, disample);
    else
        % CNN, GSNN
        [cost, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, KL] = ...
            cnn_cost(x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
            predmodel, gradcheck, dropout, doder, disample);
    end
end

return;



% -----------------------------------------------------------------------
%   cost and gradient function of CNN or GSNN
%   (i.e., CNN + stochastic neuron)
%
%   supports: multi-scale, flat encoding
%
%   x   : input data
%   target data should have been fed to net_cls.layers{end}.data
% -----------------------------------------------------------------------

function [cost, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, KL] = ...
    cnn_cost(x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, predmodel, gradcheck, dropout, doder, disample)

if ~exist('predmodel', 'var'),
    predmodel = 'pyxz';
end

[cost, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, KL] = cnn_cost_sub...
    (x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, predmodel, gradcheck, dropout, doder, disample);

return;

function [cost, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, KL] = cnn_cost_sub...
    (x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, predmodel, gradcheck, dropout, doder, disample)

if ~exist('gradcheck', 'var'),
    gradcheck = 0;
end
if ~exist('dropout', 'var'),
    dropout = 0;
end
if ~exist('disample', 'var'),
    disample = 0;
end
if ~exist('doder', 'var'),
    doder = 1;
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

cost = 0;
KL = 0;
nscale = length(net_cls);


% -------------------------------------------------------------------------
%                                                     generation (x -> y)
% -------------------------------------------------------------------------

res_x2y{1} = mcn_ff(net_x2y{1}, x, [], res_x2y{1}, opts);
for j = 2:nscale,
    res_x2y{j} = mcn_ff(net_x2y{j}, res_x2y{j-1}(end).x, [], res_x2y{j}, opts);
end
res_cnn{nscale} = mcn_ff(net_cnn{nscale}, res_x2y{nscale}(end).x, [], res_cnn{nscale}, opts);

for j = 1:nscale-1,
    res_cnn{j} = mcn_ff(net_cnn{j}, res_x2y{j}(end).x, [], res_cnn{j}, opts);
    res_cls{j} = mcn_ff(net_cls{j}, res_cnn{j}(end).x, [], res_cnn{j}, opts);
    cost = cost + res_cls{j}(end).x;
end


% -------------------------------------------------------------------------
%                                             generation (x,y -> z -> hy)
% -------------------------------------------------------------------------

res_cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x, [], res_cls{nscale}, opts);
cost = cost + res_cls{nscale}(end).x;


% -------------------------------------------------------------------------
%                                                   generation (gradient)
% -------------------------------------------------------------------------

if doder,
    res_cls{nscale} = mcn_bp(net_cls{nscale}, one, res_cls{nscale}, opts);
    for j = nscale-1:-1:1,
        res_cls{j} = mcn_bp(net_cls{j}, one, res_cls{j}, opts);
    end
end


% -------------------------------------------------------------------------
%                                         generation (gradient) (hy -> x)
% -------------------------------------------------------------------------

if doder,
    res_cnn{nscale}(end).dzdx = res_cls{nscale}(1).dzdx;
    res_cnn{nscale}(end).dzds = res_cls{nscale}(1).dzds;
    if strcmp(predmodel, 'pyz'),
        res_cnn{nscale}(end).dzdx = 0*res_cnn{nscale}(end).dzdx;
        res_cnn{nscale}(end).dzds = 0*res_cnn{nscale}(end).dzds;
    end
    res_cnn{nscale} = mcn_bp(net_cnn{nscale}, [], res_cnn{nscale}, opts);
    
    res_x2y{nscale}(end).dzdx = res_cnn{nscale}(1).dzdx;
    res_x2y{nscale}(end).dzds = res_cnn{nscale}(1).dzds;
    res_x2y{nscale} = mcn_bp(net_x2y{nscale}, [], res_x2y{nscale}, opts);
    
    for j = nscale-1:-1:1,
        res_cnn{j}(end).dzdx = res_cls{j}(1).dzdx;
        res_cnn{j}(end).dzds = res_cls{j}(1).dzds;
        res_cnn{j} = mcn_bp(net_cnn{j}, [], res_cnn{j}, opts);
        
        res_x2y{j}(end).dzdx = res_x2y{j+1}(1).dzdx + res_cnn{j}(1).dzdx;
        res_x2y{j}(end).dzds = res_cnn{j}(1).dzds;
        res_x2y{j} = mcn_bp(net_x2y{j}, [], res_x2y{j}, opts);
    end
end

return;



% -----------------------------------------------------------------------
%   cost and gradient function of CVAE or GSNN
%
%   supports: multi-scale, flat encoding
%
%   x   : input data
%   y   : output data (original scale)
%   multi-scale output should have been fed to net_cls.layers{end}.data
% -----------------------------------------------------------------------

function [cost, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
    cvae_cost(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
    alpha, recmodel, predmodel, gradcheck, dropout, doder, disample)

if ~exist('alpha', 'var'),
    alpha = 0;
end
if ~exist('recmodel', 'var'),
    recmodel = 'qzxy';
end
if ~exist('predmodel', 'var'),
    predmodel = 'pyxz';
end

[cost, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
    cvae_cost_sub(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
    alpha, recmodel, predmodel, gradcheck, dropout, doder, disample);

return;

function [cost, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
    cvae_cost_sub(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
    alpha, recmodel, predmodel, gradcheck, dropout, doder, disample)

if ~exist('alpha', 'var'),
    alpha = 1;
end
if ~exist('gradcheck', 'var'),
    gradcheck = 0;
end
if ~exist('dropout', 'var'),
    dropout = 0;
end
if ~exist('doder', 'var'),
    doder = 1;
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

cost = 0;
KL = 0;
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

for j = 1:nscale-1,
    res_cnn{j} = mcn_ff(net_cnn{j}, res_x2y{j}(end).x, [], res_cnn{j}, opts);
    res_cls{j} = mcn_ff(net_cls{j}, res_cnn{j}(end).x, [], res_cnn{j}, opts);
    cost = cost + res_cls{j}(end).x;
end


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
%                                                 KL divergence loss
% -----------------------------------------------------------------------

s1sq = res_xy2z(end).s.^2;
s2sq = res_x2z(end).s.^2;
tmp = (s1sq + (res_xy2z(end).x - res_x2z(end).x).^2)./(s2sq);
KL = KL - 0.5*sum(sum(sum(sum(one + 2*log(res_xy2z(end).s), 1), 2), 3), 4);
KL = KL + 0.5*sum(sum(sum(sum((2*log(res_x2z(end).s) + tmp), 1), 2), 3), 4);
cost = cost + alpha*KL;


% -----------------------------------------------------------------------
%                                                generation (z -> y)
% -----------------------------------------------------------------------

% prediction (used for gradient computation)
res_x2z2y = mcn_ff(net_z2y, res_x2z(end).x, res_x2z(end).s, res_x2z2y, opts);
if strcmp(predmodel, 'pyxz'),
    res_x2cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x + res_x2z2y(end).x, [], res_x2cls{nscale}, opts);
elseif strcmp(predmodel, 'pyz'),
    res_x2cls{nscale} = mcn_ff(net_cls{nscale}, res_x2z2y(end).x, [], res_x2cls{nscale}, opts);
end
cost = cost + (one-alpha)*res_x2cls{nscale}(end).x;

% actual prediction result
c = res_x2cls{nscale}(end).x;
p = res_x2cls{nscale}(end-1).x;
ps = res_x2cls{nscale}(end-1).s;

% generation (used for gradient computation)
res_z2y = mcn_ff(net_z2y, res_xy2z(end).x, res_xy2z(end).s, res_z2y, opts);
if strcmp(predmodel, 'pyxz'),
    res_cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x + res_z2y(end).x, [], res_cls{nscale}, opts);
elseif strcmp(predmodel, 'pyz'),
    res_cls{nscale} = mcn_ff(net_cls{nscale}, res_z2y(end).x, [], res_cls{nscale}, opts);
end
cost = cost + alpha*res_cls{nscale}(end).x;


if doder,
    
    % -------------------------------------------------------------------
    %                                          generation (gradient)
    % -------------------------------------------------------------------
    
    res_cls{nscale} = mcn_bp(net_cls{nscale}, alpha, res_cls{nscale}, opts);
    for j = nscale-1:-1:1,
        res_cls{j} = mcn_bp(net_cls{j}, one, res_cls{j}, opts);
    end
    res_z2y(end).dzdx = res_cls{nscale}(1).dzdx;
    res_z2y(end).dzds = res_cls{nscale}(1).dzds;
    res_z2y = mcn_bp(net_z2y, [], res_z2y, opts);
    
    res_x2cls{nscale} = mcn_bp(net_cls{nscale}, one-alpha, res_x2cls{nscale}, opts);
    res_x2z2y(end).dzdx = res_x2cls{nscale}(1).dzdx;
    res_x2z2y(end).dzds = res_x2cls{nscale}(1).dzds;
    res_x2z2y = mcn_bp(net_z2y, [], res_x2z2y, opts);
    
    
    % -------------------------------------------------------------------
    %                                 generation (gradient) (z -> x)
    % -------------------------------------------------------------------
    
    res_x2z(end).dzdx = res_x2z2y(1).dzdx - alpha*(res_xy2z(end).x - res_x2z(end).x)./s2sq;
    res_x2z(end).dzds = res_x2z2y(1).dzds + alpha*0.5*(1 - tmp);
    res_x2z = mcn_bp(net_x2z, [], res_x2z, opts);
    
    
    % -------------------------------------------------------------------
    %                              recognition (gradient) (z -> x,y)
    % -------------------------------------------------------------------
    
    res_xy2z(end).dzdx = res_z2y(1).dzdx + alpha*(res_xy2z(end).x - res_x2z(end).x)./s2sq;
    res_xy2z(end).dzds = res_z2y(1).dzds - alpha*0.5*(1 - s1sq./s2sq);
    res_xy2z = mcn_bp(net_xy2z, [], res_xy2z, opts);
    
    
    % -------------------------------------------------------------------
    %                                generation (gradient) (hy -> x)
    % -------------------------------------------------------------------
    
    res_cnn{nscale}(end).dzdx = res_x2cls{nscale}(1).dzdx + res_cls{nscale}(1).dzdx;
    res_cnn{nscale}(end).dzds = res_x2cls{nscale}(1).dzds + res_cls{nscale}(1).dzds;
    if strcmp(predmodel, 'pyz'),
        res_cnn{nscale}(end).dzdx = 0*res_cnn{nscale}(end).dzdx;
        res_cnn{nscale}(end).dzds = 0*res_cnn{nscale}(end).dzds;
    end
    res_cnn{nscale} = mcn_bp(net_cnn{nscale}, [], res_cnn{nscale}, opts);
    
    res_x2y{nscale}(end).dzdx = res_cnn{nscale}(1).dzdx;
    res_x2y{nscale}(end).dzds = res_cnn{nscale}(1).dzds;
    res_x2y{nscale} = mcn_bp(net_x2y{nscale}, [], res_x2y{nscale}, opts);
    
    for j = nscale-1:-1:1,
        res_cnn{j}(end).dzdx = res_cls{j}(1).dzdx;
        res_cnn{j}(end).dzds = res_cls{j}(1).dzds;
        res_cnn{j} = mcn_bp(net_cnn{j}, [], res_cnn{j}, opts);
        
        res_x2y{j}(end).dzdx = res_x2y{j+1}(1).dzdx + res_cnn{j}(1).dzdx;
        res_x2y{j}(end).dzds = res_cnn{j}(1).dzds;
        res_x2y{j} = mcn_bp(net_x2y{j}, [], res_x2y{j}, opts);
    end
    
    % add up gradients
    res_z2y = mcn_add_struct(res_z2y, res_x2z2y);
    res_cls{nscale} = mcn_add_struct(res_cls{nscale}, res_x2cls{nscale});
end

res_cls{nscale}(end).x = c;
res_cls{nscale}(end-1).x = p;
res_cls{nscale}(end-1).s = ps;

return;



% -----------------------------------------------------------------------
%   cost and gradient function of CNN or GSNN
%   (i.e., CNN + stochastic neuron)
%
%   supports: multi-scale, recurrent encoding
%
%   x   : input data
%   target data is should have been fed to nec_cls.layers{end}.data
% -----------------------------------------------------------------------

function [cost, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
    cnn_recurrent_cost(x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
    genmodel, predmodel, gradcheck, dropout, doder, disample)

if ~exist('genmodel', 'var'),
    genmodel = 'pzx';
end
if ~exist('predmodel', 'var'),
    predmodel = 'pyxz';
end

[cost, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2cls, KL] = cnn_recurrent_cost_sub...
    (x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2cls, ...
    genmodel, predmodel, gradcheck, dropout, doder, disample);

return;

function [cost, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2cls, KL] = cnn_recurrent_cost_sub...
    (x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2cls, ...
    genmodel, predmodel, gradcheck, dropout, doder, disample)

if ~exist('gradcheck', 'var'),
    gradcheck = 0;
end
if ~exist('dropout', 'var'),
    dropout = 0;
end
if ~exist('disample', 'var'),
    disample = 0;
end
if ~exist('doder', 'var'),
    doder = 1;
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

cost = 0;
KL = 0;
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

for j = 1:nscale-1,
    res_cnn{j} = mcn_ff(net_cnn{j}, res_x2y{j}(end).x, [], res_cnn{j}, opts);
    res_cls{j} = mcn_ff(net_cls{j}, res_cnn{j}(end).x, [], res_cnn{j}, opts);
    cost = cost + res_cls{j}(end).x;
end


% -----------------------------------------------------------------------
%                                       generation (x,y -> z -> hy)
% -----------------------------------------------------------------------

if strcmp(genmodel, 'pzx'),
    res_x2z = mcn_ff(net_x2z, cat(3, x, res_cls{nscale}(1).x), [], res_x2z, opts);
elseif strcmp(genmodel, 'pzy'),
    res_x2z = mcn_ff(net_x2z, res_cls{nscale}(1).x, [], res_x2z, opts);
end
res_z2y = mcn_ff(net_z2y, res_x2z(end).x, res_x2z(end).s, res_z2y, opts);

if strcmp(predmodel, 'pyxz'),
    res_x2cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x + res_z2y(end).x, [], res_x2cls{nscale}, opts);
elseif strcmp(predmodel, 'pyz'),
    res_x2cls{nscale} = mcn_ff(net_cls{nscale}, res_z2y(end).x, [], res_x2cls{nscale}, opts);
end
cost = cost + res_x2cls{nscale}(end).x;

c = res_x2cls{nscale}(end).x;
p = res_x2cls{nscale}(end-1).x;
ps = res_x2cls{nscale}(end-1).s;


% -----------------------------------------------------------------------
%                                               generation (gradient)
% -----------------------------------------------------------------------

if doder,
    res_x2cls{nscale} = mcn_bp(net_cls{nscale}, one, res_x2cls{nscale}, opts);
    for j = nscale-1:-1:1,
        res_cls{j} = mcn_bp(net_cls{j}, one, res_cls{j}, opts);
    end
    res_z2y(end).dzdx = res_x2cls{nscale}(1).dzdx;
    res_z2y(end).dzds = res_x2cls{nscale}(1).dzds;
    res_z2y = mcn_bp(net_z2y, [], res_z2y, opts);
    
    res_x2z(end).dzdx = res_z2y(1).dzdx;
    res_x2z(end).dzds = res_z2y(1).dzds;
    res_x2z = mcn_bp(net_x2z, [], res_x2z, opts);
end


% -----------------------------------------------------------------------
%                                    generation (gradient) (hy -> x)
% -----------------------------------------------------------------------

if doder,
    res_cnn{nscale}(end).dzdx = res_x2cls{nscale}(1).dzdx;
    res_cnn{nscale}(end).dzds = res_x2cls{nscale}(1).dzds;
    
    % recurrent nets
    net_cls{nscale}.layers{end}.type = strtok(net_cls{nscale}.layers{end}.type, '_');
    res_cls{nscale} = mcn_ff(net_cls{nscale}, scale_factor*res_cnn{nscale}(end).x, [], res_cls{nscale}, opts);
    
    res_cls{nscale}(end).dzdx = res_x2z(1).dzdx(:,:,end-size(res_cls{nscale}(end).x,3)+1:end,:);
    res_cls{nscale}(end).dzds = res_x2z(1).dzds(:,:,end-size(res_cls{nscale}(end).s,3)+1:end,:);
    res_cls{nscale} = mcn_bp(net_cls{nscale}, [], res_cls{nscale}, opts);
    
    if strcmp(predmodel, 'pyxz'),
        res_cnn{nscale}(end).dzdx = res_cnn{nscale}(end).dzdx + scale_factor*res_cls{nscale}(1).dzdx;
        res_cnn{nscale}(end).dzds = res_cnn{nscale}(end).dzds + scale_factor*res_cls{nscale}(1).dzds;
    elseif strcmp(predmodel, 'pyz'),
        res_cnn{nscale}(end).dzdx = scale_factor*res_cls{nscale}(1).dzdx;
        res_cnn{nscale}(end).dzds = scale_factor*res_cls{nscale}(1).dzds;
    end
    res_cnn{nscale} = mcn_bp(net_cnn{nscale}, [], res_cnn{nscale}, opts);
    
    res_x2y{nscale}(end).dzdx = res_cnn{nscale}(1).dzdx;
    res_x2y{nscale}(end).dzds = res_cnn{nscale}(1).dzds;
    res_x2y{nscale} = mcn_bp(net_x2y{nscale}, [], res_x2y{nscale}, opts);
    
    for j = nscale-1:-1:1,
        res_cnn{j}(end).dzdx = res_cls{j}(1).dzdx;
        res_cnn{j}(end).dzds = res_cls{j}(1).dzds;
        res_cnn{j} = mcn_bp(net_cnn{j}, [], res_cnn{j}, opts);
        
        res_x2y{j}(end).dzdx = res_x2y{j+1}(1).dzdx + res_cnn{j}(1).dzdx;
        res_x2y{j}(end).dzds = res_cnn{j}(1).dzds;
        res_x2y{j} = mcn_bp(net_x2y{j}, [], res_x2y{j}, opts);
    end
    
    % add gradient
    res_cls{nscale} = mcn_add_struct(res_cls{nscale}, res_x2cls{nscale});
    res_cls{nscale}(end).x = res_x2cls{nscale}(end).x;
end

res_cls{nscale}(end).x = c;
res_cls{nscale}(end-1).x = p;
res_cls{nscale}(end-1).s = ps;

return;



% -----------------------------------------------------------------------
%   cost and gradient function of CVAE or GSNN
%
%   supports: multi-scale, recurrent encoding
%
%   x   : input data
%   y   : output data (original scale)
%   multi-scale output should have been fed to net_cls.layers{end}.data
% -----------------------------------------------------------------------

function [cost, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
    cvae_recurrent_cost(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
    alpha, recmodel, genmodel, predmodel, gradcheck, dropout, doder, disample)

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

[cost, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
    cvae_recurrent_cost_sub(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
    alpha, recmodel, genmodel, predmodel, gradcheck, dropout, doder, disample);

return;

function [cost, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
    cvae_recurrent_cost_sub(x, y, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
    alpha, recmodel, genmodel, predmodel, gradcheck, dropout, doder, disample)

if ~exist('alpha', 'var'),
    alpha = 1;
end
if ~exist('gradcheck', 'var'),
    gradcheck = 0;
end
if ~exist('dropout', 'var'),
    dropout = 0;
end
if ~exist('doder', 'var'),
    doder = 1;
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

cost = 0;
KL = 0;
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

for j = 1:nscale-1,
    res_cnn{j} = mcn_ff(net_cnn{j}, res_x2y{j}(end).x, [], res_cnn{j}, opts);
    res_cls{j} = mcn_ff(net_cls{j}, res_cnn{j}(end).x, [], res_cnn{j}, opts);
    cost = cost + res_cls{j}(end).x;
end


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
%                                                KL divergence loss
% -----------------------------------------------------------------------

s1sq = res_xy2z(end).s.^2;
s2sq = res_x2z(end).s.^2;
tmp = (s1sq + (res_xy2z(end).x - res_x2z(end).x).^2)./(s2sq);
KL = KL - 0.5*sum(sum(sum(sum(one + 2*log(res_xy2z(end).s), 1), 2), 3), 4);
KL = KL + 0.5*sum(sum(sum(sum((2*log(res_x2z(end).s) + tmp), 1), 2), 3), 4);
cost = cost + alpha*KL;


% -----------------------------------------------------------------------
%                                                generation (z -> y)
% -----------------------------------------------------------------------

% prediction (used for gradient computation)
res_x2z2y = mcn_ff(net_z2y, res_x2z(end).x, res_x2z(end).s, res_x2z2y, opts);
if strcmp(predmodel, 'pyxz'),
    res_x2cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x + res_x2z2y(end).x, [], res_x2cls{nscale}, opts);
elseif strcmp(predmodel, 'pyz'),
    res_x2cls{nscale} = mcn_ff(net_cls{nscale}, res_x2z2y(end).x, [], res_x2cls{nscale}, opts);
end
cost = cost + (one-alpha)*res_x2cls{nscale}(end).x;

% actual prediction result
c = res_x2cls{nscale}(end).x;
p = res_x2cls{nscale}(end-1).x;
ps = res_x2cls{nscale}(end-1).s;

% generation (used for gradient computation)
res_z2y = mcn_ff(net_z2y, res_xy2z(end).x, res_xy2z(end).s, res_z2y, opts);
if strcmp(predmodel, 'pyxz'),
    res_cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x + res_z2y(end).x, [], res_cls{nscale}, opts);
elseif strcmp(predmodel, 'pyz'),
    res_cls{nscale} = mcn_ff(net_cls{nscale}, res_z2y(end).x, [], res_cls{nscale}, opts);
end
cost = cost + alpha*res_cls{nscale}(end).x;


if doder,
    
    % -----------------------------------------------------------------------
    %                                                   generation (gradient)
    % -----------------------------------------------------------------------
    
    res_cls{nscale} = mcn_bp(net_cls{nscale}, alpha, res_cls{nscale}, opts);
    for j = nscale-1:-1:1,
        res_cls{j} = mcn_bp(net_cls{j}, one, res_cls{j}, opts);
    end
    res_z2y(end).dzdx = res_cls{nscale}(1).dzdx;
    res_z2y(end).dzds = res_cls{nscale}(1).dzds;
    res_z2y = mcn_bp(net_z2y, [], res_z2y, opts);
    
    res_x2cls{nscale} = mcn_bp(net_cls{nscale}, one-alpha, res_x2cls{nscale}, opts);
    res_x2z2y(end).dzdx = res_x2cls{nscale}(1).dzdx;
    res_x2z2y(end).dzds = res_x2cls{nscale}(1).dzds;
    res_x2z2y = mcn_bp(net_z2y, [], res_x2z2y, opts);
    
    
    % -----------------------------------------------------------------------
    %                                          generation (gradient) (z -> x)
    % -----------------------------------------------------------------------
    
    res_x2z(end).dzdx = res_x2z2y(1).dzdx - alpha*(res_xy2z(end).x - res_x2z(end).x)./s2sq;
    res_x2z(end).dzds = res_x2z2y(1).dzds + alpha*0.5*(1 - tmp);
    res_x2z = mcn_bp(net_x2z, [], res_x2z, opts);
    
    
    % -----------------------------------------------------------------------
    %                                       recognition (gradient) (z -> x,y)
    % -----------------------------------------------------------------------
    
    res_xy2z(end).dzdx = res_z2y(1).dzdx + alpha*(res_xy2z(end).x - res_x2z(end).x)./s2sq;
    res_xy2z(end).dzds = res_z2y(1).dzds - alpha*0.5*(1 - s1sq./s2sq);
    res_xy2z = mcn_bp(net_xy2z, [], res_xy2z, opts);
    
    
    % -----------------------------------------------------------------------
    %                                         generation (gradient) (hy -> x)
    % -----------------------------------------------------------------------
    
    res_cnn{nscale}(end).dzdx = res_x2cls{nscale}(1).dzdx + res_cls{nscale}(1).dzdx;
    res_cnn{nscale}(end).dzds = res_x2cls{nscale}(1).dzds + res_cls{nscale}(1).dzds;
    if strcmp(predmodel, 'pyz'),
        res_cnn{nscale}(end).dzdx = 0*res_cnn{nscale}(end).dzdx;
        res_cnn{nscale}(end).dzds = 0*res_cnn{nscale}(end).dzds;
    end
    res_cnn{nscale} = mcn_bp(net_cnn{nscale}, [], res_cnn{nscale}, opts);
    
    % recurrent nets
    res_cnn{nscale} = mcn_ff(net_cnn{nscale}, res_x2y{nscale}(end).x, res_x2y{nscale}(end).s, res_cnn{nscale}, opts);
    res_cls{nscale} = mcn_ff(net_cls{nscale}, scale_factor*res_cnn{nscale}(end).x, [], res_cls{nscale}, opts);
    net_cls{nscale}.layers{end}.type = strtok(net_cls{nscale}.layers{end}.type, '_');
    res_cls{nscale} = mcn_ff(net_cls{nscale}, scale_factor*res_cnn{nscale}(end).x, scale_factor*res_cnn{nscale}(end).s, res_cls{nscale}, opts);
    
    res_cls{nscale}(end).dzdx = res_x2z(1).dzdx(:,:,end-size(res_cls{nscale}(end).x,3)+1:end,:);
    res_cls{nscale}(end).dzds = res_x2z(1).dzds(:,:,end-size(res_cls{nscale}(end).s,3)+1:end,:);
    res_cls{nscale} = mcn_bp(net_cls{nscale}, [], res_cls{nscale}, opts);
    
    res_cnn{nscale}(end).dzdx = res_cnn{nscale}(end).dzdx + scale_factor*res_cls{nscale}(1).dzdx;
    res_cnn{nscale}(end).dzds = res_cnn{nscale}(end).dzds + scale_factor*res_cls{nscale}(1).dzds;
    res_cnn{nscale} = mcn_bp(net_cnn{nscale}, [], res_cnn{nscale}, opts);
    
    res_x2y{nscale}(end).dzdx = res_cnn{nscale}(1).dzdx;
    res_x2y{nscale}(end).dzds = res_cnn{nscale}(1).dzds;
    res_x2y{nscale} = mcn_bp(net_x2y{nscale}, [], res_x2y{nscale}, opts);
    
    for j = nscale-1:-1:1,
        res_cnn{j}(end).dzdx = res_cls{j}(1).dzdx;
        res_cnn{j}(end).dzds = res_cls{j}(1).dzds;
        res_cnn{j} = mcn_bp(net_cnn{j}, [], res_cnn{j}, opts);
        
        res_x2y{j}(end).dzdx = res_x2y{j+1}(1).dzdx + res_cnn{j}(1).dzdx;
        res_x2y{j}(end).dzds = res_cnn{j}(1).dzds;
        res_x2y{j} = mcn_bp(net_x2y{j}, [], res_x2y{j}, opts);
    end
    
    % add up gradients
    res_z2y = mcn_add_struct(res_z2y, res_x2z2y);
    res_cls{nscale} = mcn_add_struct(res_cls{nscale}, res_x2cls{nscale});
end

res_cls{nscale}(end).x = c;
res_cls{nscale}(end-1).x = p;
res_cls{nscale}(end-1).s = ps;

return;

