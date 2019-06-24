function [res_z2y, res_x2z, res_x2y, res_cnn, res_cls, pred_cnn, pred_gen] = ...
    dcgm_inference(x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, do_recurrent, genmodel, predmodel)

if ~exist('do_recurrent', 'var'),
    do_recurrent = 0;
end
if ~exist('genmodel', 'var'),
    genmodel = 'pzx';
end
if ~exist('predmodel', 'var'),
    predmodel = 'pyxz';
end

dropout = 0;
disample = 0;

if do_recurrent,
    % recurrent encoding
    % hybrid, CVAE, CNN, GSNN
    if nargout > 5,
        [res_z2y, res_x2z, res_x2y, res_cnn, res_cls, pred_cnn, pred_gen] = dcgm_recurrent_inf...
            (x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_z2y, res_x2z, res_x2y, res_cnn, res_cls, genmodel, predmodel, dropout, disample);
    else
        [res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = dcgm_recurrent_inf...
            (x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_z2y, res_x2z, res_x2y, res_cnn, res_cls, genmodel, predmodel, dropout, disample);
    end
else
    % hybrid, CVAE, CNN, GSNN
    [res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = dcgm_inf...
        (x, net_x2y, net_cnn, net_cls, res_z2y, res_x2z, res_x2y, ...
        res_cnn, res_cls, predmodel, dropout, disample);
    pred_cnn = [];
    pred_gen = [];
end

return;



% -----------------------------------------------------------------------
%   inference function of CVAE or GSNN
%
%   supports: multi-scale, flat encoding
%
%   x   : input data
%   output can be found at res_cls{nscale}(end).x
% -----------------------------------------------------------------------

function [res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = dcgm_inf...
    (x, net_x2y, net_cnn, net_cls, res_z2y, res_x2z, res_x2y, ...
    res_cnn, res_cls, predmodel, dropout, disample)

if ~exist('dropout', 'var'),
    dropout = 0;
end
if ~exist('disample', 'var'),
    disample = 0;
end
if ~exist('predmodel', 'var'),
    predmodel = 'pyxz';
end

opts = struct(...
    'gradcheck', 0, ...
    'disableDropout', dropout, ...
    'disableSampling', disample);

nscale = length(net_cls);


% -----------------------------------------------------------------------
%                                                generation (x -> y)
% -----------------------------------------------------------------------

opts_x2y = opts;
opts_x2y.disableSampling = 1;
res_x2y{1} = mcn_ff(net_x2y{1}, x, [], res_x2y{1}, opts_x2y);
for j = 2:nscale,
    res_x2y{j} = mcn_ff(net_x2y{j}, res_x2y{j-1}(end).x, res_x2y{j-1}(end).s, res_x2y{j}, opts_x2y);
end

res_cnn{nscale} = mcn_ff(net_cnn{nscale}, res_x2y{nscale}(end).x, res_x2y{nscale}(end).s, res_cnn{nscale}, opts_x2y);


% -----------------------------------------------------------------------
%                                               generation (hy -> y)
% -----------------------------------------------------------------------

if strcmp(predmodel, 'pyxz'),
    res_cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x, [], res_cls{nscale}, opts);
elseif strcmp(predmodel, 'pyz'),
    res_cls{nscale} = mcn_ff(net_cls{nscale}, res_z2y(end).x, [], res_cls{nscale}, opts);
end

return;



% -----------------------------------------------------------------------
%   inference function of CVAE or GSNN
%
%   supports: multi-scale, recurrent encoding
%
%   x   : input data
%   output can be found at res_cls{nscale}(end).x
% -----------------------------------------------------------------------

function [res_z2y, res_x2z, res_x2y, res_cnn, res_cls, pred_cnn, pred_gen] = dcgm_recurrent_inf...
    (x, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, genmodel, predmodel, dropout, disample)

if ~exist('dropout', 'var'),
    dropout = 0;
end
if ~exist('disample', 'var'),
    disample = 0;
end
if ~exist('genmodel', 'var'),
    genmodel = 'pzx';
end
if ~exist('predmodel', 'var'),
    predmodel = 'pyxz';
end

opts = struct(...
    'gradcheck', 0, ...
    'disableDropout', dropout, ...
    'disableSampling', disample);

nscale = length(net_cls);


% -----------------------------------------------------------------------
%                                                generation (x -> y)
% -----------------------------------------------------------------------

opts_x2y = opts;
opts_x2y.disableSampling = 1;
res_x2y{1} = mcn_ff(net_x2y{1}, x, [], res_x2y{1}, opts_x2y);
for j = 2:nscale,
    res_x2y{j} = mcn_ff(net_x2y{j}, res_x2y{j-1}(end).x, res_x2y{j-1}(end).s, res_x2y{j}, opts_x2y);
end

res_cnn{nscale} = mcn_ff(net_cnn{nscale}, res_x2y{nscale}(end).x, res_x2y{nscale}(end).s, res_cnn{nscale}, opts_x2y);
if strcmp(predmodel, 'pyxz'),
    scale_factor = 2;
elseif strcmp(predmodel, 'pyz'),
    scale_factor = 1;
end
res_cls{nscale} = mcn_ff(net_cls{nscale}, scale_factor*res_cnn{nscale}(end).x, scale_factor*res_cnn{nscale}(end).s, res_cls{nscale}, opts_x2y);
if nargout >= 6,
    pred_cnn = res_cls{nscale}(end-1).x;
end


% -----------------------------------------------------------------------
%                                                generation (x -> z)
% -----------------------------------------------------------------------

if strcmp(genmodel, 'pzx'),
    res_x2z = mcn_ff(net_x2z, cat(3, x, res_cls{nscale}(1).x), [], res_x2z, opts);
elseif strcmp(genmodel, 'pzy'),
    res_x2z = mcn_ff(net_x2z, res_cls{nscale}(1).x, [], res_x2z, opts);
end


% -----------------------------------------------------------------------
%                                               generation (z -> hy)
% -----------------------------------------------------------------------

opts_z2y = opts;
opts_z2y.disableSampling = 1;
res_z2y = mcn_ff(net_z2y, res_x2z(end).x, res_x2z(end).s, res_z2y, opts_z2y);


% -----------------------------------------------------------------------
%                                               generation (hy -> y)
% -----------------------------------------------------------------------

if nargout == 7,
    res_cls{nscale} = mcn_ff(net_cls{nscale}, scale_factor*res_z2y(end).x, [], res_cls{nscale}, opts);
    pred_gen = res_cls{nscale}(end-1).x;
end

if strcmp(predmodel, 'pyxz'),
    res_cls{nscale} = mcn_ff(net_cls{nscale}, res_cnn{nscale}(end).x + res_z2y(end).x, [], res_cls{nscale}, opts);
elseif strcmp(predmodel, 'pyz'),
    res_cls{nscale} = mcn_ff(net_cls{nscale}, res_z2y(end).x, [], res_cls{nscale}, opts);
end

return;
