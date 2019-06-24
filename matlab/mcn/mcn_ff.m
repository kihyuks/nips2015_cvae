% -----------------------------------------------------------------------
%   matConvNet -- reformatted to support two input variables, x and s,
%   (forward)     for gaussian random variable.
%
% -----------------------------------------------------------------------

function [res, cost] = mcn_ff(net, x, s, res, varargin)


% -----------------------------------------------------------------------
%                                                  default parameters
% -----------------------------------------------------------------------

opts.res = [];
opts.conserveMemory = false;
opts.sync = false;
opts.disableDropout = false;
opts.freezeDropout = false;
opts.disableSampling = false;
opts.gradcheck = false;
opts.visualize = false;
opts.ffLayer_init = 1;
opts.ffLayer_end = numel(net.layers);

opts = vl_argparse(opts, varargin);


% -----------------------------------------------------------------------
%                                                      initialization
% -----------------------------------------------------------------------

nlayer = numel(net.layers);
gpuMode = isa(x, 'gpuArray');
if opts.gradcheck,
    % gradient check
    one = double(1);
    zero = double(0);
    gpuMode = 0;
else
    one = single(1);
    zero = single(0);
end
if gpuMode,
    one = gpuArray(one);
    zero = gpuArray(zero);
end

if ~exist('res', 'var') || isempty(res)
    res = struct(...
        'x', cell(1, nlayer+1), ...
        's', cell(1, nlayer+1), ...
        'dzdx', cell(1, nlayer+1), ...
        'dzds', cell(1, nlayer+1), ...
        'filters', cell(1, nlayer+1), ...
        'biases', cell(1, nlayer+1), ...
        'filters_std', cell(1, nlayer+1), ...
        'biases_std', cell(1, nlayer+1), ...
        'aux', cell(1, nlayer+1), ...
        'time', num2cell(zeros(1, nlayer+1)), ...
        'backwardTime', num2cell(zeros(1, nlayer+1)));
end

if ~isempty(x) && isempty(s),
    res(1).x = x;
end
if ~isempty(x) && ~isempty(s),
    res(1).x = x;
    res(1).s = s;
end

cost = zero;


% -----------------------------------------------------------------------
%                                                 forward propagation
% 1. linear response    : conv
% 2. sampling           : gaussian_sampling
% 3. pooling            : pool, unpool
% 4. nonlinearities     : relu, sigmoid, softplus, softmax, normalize
% 5. loss               : bernoulli_loss, gaussian_loss, softmax_loss,
%                         kldiv_loss (to be minimized)
% 6. dropout            : dropout
% -----------------------------------------------------------------------

for i = opts.ffLayer_init:opts.ffLayer_end,
    l = net.layers{i};
    res(i).time = tic;
    
    switch l.type
        case 'conv',
            % linear (convolutional) response
            if opts.gradcheck,
                if length(l.pad) > 1 && sum(l.pad) > 0,
                    x = zeros(size(res(i).x, 1) + l.pad(1) + l.pad(3), size(res(i).x, 2) + l.pad(2) + l.pad(4), size(res(i).x, 3), size(res(i).x, 4));
                    x(l.pad(1)+1:l.pad(1)+size(res(i).x, 1), l.pad(2)+1:l.pad(2)+size(res(i).x, 2), :, :) = res(i).x;
                    res(i+1).x = zeros(size(x, 1)-size(l.filters, 1)+1, size(x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                else
                    res(i+1).x = zeros(size(res(i).x, 1)-size(l.filters, 1)+1, size(res(i).x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(res(i).x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(res(i).x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                end
            else
                res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride);
            end
            
        case 'conv_valid',
            % reshape followed by linear response (valid convolution)
            dimx = size(res(i).x);
            dimf = l.fdim;
            res(i).x = reshape(res(i).x, dimx(1)./dimf(1), dimx(2)./dimf(2), dimf(1)*dimf(2)*dimx(3), dimx(4));
            if opts.gradcheck,
                if length(l.pad) > 1 && sum(l.pad) > 0,
                    x = zeros(size(res(i).x, 1) + l.pad(1) + l.pad(3), size(res(i).x, 2) + l.pad(2) + l.pad(4), size(res(i).x, 3), size(res(i).x, 4));
                    x(l.pad(1)+1:l.pad(1)+size(res(i).x, 1), l.pad(2)+1:l.pad(2)+size(res(i).x, 2), :, :) = res(i).x;
                    res(i+1).x = zeros(size(x, 1)-size(l.filters, 1)+1, size(x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                else
                    res(i+1).x = zeros(size(res(i).x, 1)-size(l.filters, 1)+1, size(res(i).x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(res(i).x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(res(i).x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                end
            else
                res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride);
            end
            
        case 'conv_full',
            % reshape followed by linear response (full convolution)
            if opts.gradcheck,
                if length(l.pad) > 1 && sum(l.pad) > 0,
                    x = zeros(size(res(i).x, 1) + l.pad(1) + l.pad(3), size(res(i).x, 2) + l.pad(2) + l.pad(4), size(res(i).x, 3), size(res(i).x, 4));
                    x(l.pad(1)+1:l.pad(1)+size(res(i).x, 1), l.pad(2)+1:l.pad(2)+size(res(i).x, 2), :, :) = res(i).x;
                    res(i+1).x = zeros(size(x, 1)-size(l.filters, 1)+1, size(x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                else
                    res(i+1).x = zeros(size(res(i).x, 1)-size(l.filters, 1)+1, size(res(i).x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(res(i).x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(res(i).x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                end
            else
                res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride);
            end
            dimx = size(res(i+1).x);
            dimf = l.fdim;
            res(i+1).x = reshape(res(i+1).x, dimf(1), dimf(2), dimx(3)./(dimf(1)*dimf(2)), dimx(4));
            
        case 'conv_gaussian',
            if ~isfield(l, 'share_std'),
                l.share_std = 0;
            end
            
            if l.share_std == 1, % spatial only
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, 1]);
            elseif l.share_std == 2, % spatial + channel
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, size(l.filters, 4)]);
            end
            
            if opts.gradcheck,
                if length(l.pad) > 1 && sum(l.pad) > 0,
                    x = zeros(size(res(i).x, 1) + l.pad(1) + l.pad(3), size(res(i).x, 2) + l.pad(2) + l.pad(4), size(res(i).x, 3), size(res(i).x, 4));
                    x(l.pad(1)+1:l.pad(1)+size(res(i).x, 1), l.pad(2)+1:l.pad(2)+size(res(i).x, 2), :, :) = res(i).x;
                    
                    % linear response (mean)
                    res(i+1).x = zeros(size(x, 1)-size(l.filters, 1)+1, size(x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                    
                    % linear response (std)
                    res(i+1).s = zeros(size(x, 1)-size(l.filters_std, 1)+1, size(x, 2)-size(l.filters_std, 2)+1, size(l.filters_std, 4), size(x, 4));
                    for nh = 1:size(l.filters_std, 4),
                        res(i+1).s(:, :, nh, :) = convn(x, l.filters_std(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases_std(nh);
                    end
                else
                    % linear response (mean)
                    res(i+1).x = zeros(size(res(i).x, 1)-size(l.filters, 1)+1, size(res(i).x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(res(i).x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(res(i).x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                    
                    % linear response (std)
                    res(i+1).s = zeros(size(res(i).x, 1)-size(l.filters_std, 1)+1, size(res(i).x, 2)-size(l.filters_std, 2)+1, size(l.filters_std, 4), size(res(i).x, 4));
                    for nh = 1:size(l.filters_std, 4),
                        res(i+1).s(:, :, nh, :) = convn(res(i).x, l.filters_std(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases_std(nh);
                    end
                end
            else
                % linear response (mean)
                res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride);
                
                % linear response (std)
                res(i+1).s = vl_nnconv(res(i).x, l.filters_std, l.biases_std, 'pad', l.pad, 'stride', l.stride);
            end
            res(i+1).s = exp(0.5*res(i+1).s);
            
        case 'conv_gaussian_valid',
            if ~isfield(l, 'share_std'),
                l.share_std = 0;
            end
            
            if l.share_std == 1, % spatial only
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, 1]);
            elseif l.share_std == 2, % spatial + channel
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, size(l.filters, 4)]);
            end
            
            % reshape before convolution
            dimx = size(res(i).x);
            dimf = l.fdim;
            res(i).x = reshape(res(i).x, dimx(1)./dimf(1), dimx(2)./dimf(2), dimf(1)*dimf(2)*dimx(3), dimx(4));
            
            if opts.gradcheck,
                if length(l.pad) > 1 && sum(l.pad) > 0,
                    x = zeros(size(res(i).x, 1) + l.pad(1) + l.pad(3), size(res(i).x, 2) + l.pad(2) + l.pad(4), size(res(i).x, 3), size(res(i).x, 4));
                    x(l.pad(1)+1:l.pad(1)+size(res(i).x, 1), l.pad(2)+1:l.pad(2)+size(res(i).x, 2), :, :) = res(i).x;
                    
                    % linear response (mean)
                    res(i+1).x = zeros(size(x, 1)-size(l.filters, 1)+1, size(x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                    
                    % linear response (std)
                    res(i+1).s = zeros(size(x, 1)-size(l.filters_std, 1)+1, size(x, 2)-size(l.filters_std, 2)+1, size(l.filters_std, 4), size(x, 4));
                    for nh = 1:size(l.filters_std, 4),
                        res(i+1).s(:, :, nh, :) = convn(x, l.filters_std(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases_std(nh);
                    end
                else
                    % linear response (mean)
                    res(i+1).x = zeros(size(res(i).x, 1)-size(l.filters, 1)+1, size(res(i).x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(res(i).x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(res(i).x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                    
                    % linear response (std)
                    res(i+1).s = zeros(size(res(i).x, 1)-size(l.filters_std, 1)+1, size(res(i).x, 2)-size(l.filters_std, 2)+1, size(l.filters_std, 4), size(res(i).x, 4));
                    for nh = 1:size(l.filters_std, 4),
                        res(i+1).s(:, :, nh, :) = convn(res(i).x, l.filters_std(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases_std(nh);
                    end
                end
            else
                % linear response (mean)
                res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride);
                
                % linear response (std)
                res(i+1).s = vl_nnconv(res(i).x, l.filters_std, l.biases_std, 'pad', l.pad, 'stride', l.stride);
            end
            res(i+1).s = exp(0.5*res(i+1).s);
            
        case 'conv_gaussian_full',
            if ~isfield(l, 'share_std'),
                l.share_std = 0;
            end
            
            if l.share_std == 1, % spatial only
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, 1]);
            elseif l.share_std == 2, % spatial + channel
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, size(l.filters, 4)]);
            end
            
            if opts.gradcheck,
                if length(l.pad) > 1 && sum(l.pad) > 0,
                    x = zeros(size(res(i).x, 1) + l.pad(1) + l.pad(3), size(res(i).x, 2) + l.pad(2) + l.pad(4), size(res(i).x, 3), size(res(i).x, 4));
                    x(l.pad(1)+1:l.pad(1)+size(res(i).x, 1), l.pad(2)+1:l.pad(2)+size(res(i).x, 2), :, :) = res(i).x;
                    
                    % linear response (mean)
                    res(i+1).x = zeros(size(x, 1)-size(l.filters, 1)+1, size(x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                    
                    % linear response (std)
                    res(i+1).s = zeros(size(x, 1)-size(l.filters_std, 1)+1, size(x, 2)-size(l.filters_std, 2)+1, size(l.filters_std, 4), size(x, 4));
                    for nh = 1:size(l.filters_std, 4),
                        res(i+1).s(:, :, nh, :) = convn(x, l.filters_std(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases_std(nh);
                    end
                else
                    % linear response (mean)
                    res(i+1).x = zeros(size(res(i).x, 1)-size(l.filters, 1)+1, size(res(i).x, 2)-size(l.filters, 2)+1, size(l.filters, 4), size(res(i).x, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i+1).x(:, :, nh, :) = convn(res(i).x, l.filters(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases(nh);
                    end
                    
                    % linear response (std)
                    res(i+1).s = zeros(size(res(i).x, 1)-size(l.filters_std, 1)+1, size(res(i).x, 2)-size(l.filters_std, 2)+1, size(l.filters_std, 4), size(res(i).x, 4));
                    for nh = 1:size(l.filters_std, 4),
                        res(i+1).s(:, :, nh, :) = convn(res(i).x, l.filters_std(end:-1:1, end:-1:1, end:-1:1, nh), 'valid') + l.biases_std(nh);
                    end
                end
            else
                % linear response (mean)
                res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride);
                
                % linear response (std)
                res(i+1).s = vl_nnconv(res(i).x, l.filters_std, l.biases_std, 'pad', l.pad, 'stride', l.stride);
            end
            res(i+1).s = exp(0.5*res(i+1).s);
            
            % reshape after convolution
            dimx = size(res(i+1).x);
            dimf = l.fdim;
            res(i+1).x = reshape(res(i+1).x, dimf(1), dimf(2), dimx(3)./(dimf(1)*dimf(2)), dimx(4));
            res(i+1).s = reshape(res(i+1).s, dimf(1), dimf(2), dimx(3)./(dimf(1)*dimf(2)), dimx(4));
            
        case 'gaussian_sampling',
            % gaussian sampling
            if opts.disableSampling,
                res(i+1).x = repmat(res(i).x, [1 1 1 l.nsample]);
            else
                if opts.gradcheck,
                    rng('default');
                    n = randn([size(res(i).s), l.nsample], 'double');
                else
                    n = randn([size(res(i).s), l.nsample], 'single');
                end
                if gpuMode,
                    res(i+1).n = gpuArray(n);
                else
                    res(i+1).n = n;
                end
                res(i+1).x = bsxfun(@plus, res(i).x, bsxfun(@times, res(i).s, res(i+1).n));
                res(i+1).x = reshape(res(i+1).x, size(res(i).s, 1), size(res(i).s, 2), size(res(i).s, 3), size(res(i).s, 4)*l.nsample);
            end
            
        case 'pool',
            % pooling
            res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method);
            
        case 'unpool',
            % unpooling
            res(i+1).x = vl_nnupsample(res(i).x, 'stride', l.stride, 'method', l.method);
            
        case {'gaussian', 'linear'},
            % linear
            res(i+1).x = res(i).x;
            res(i+1).s = res(i).s;
            
        case {'sigmoid', 'bernoulli'},
            % sigmoid
            res(i+1).x = one./(one+exp(-res(i).x));
            
        case 'relu'
            % rectified linear
            res(i+1).x = max(zero, res(i).x);
            
        case 'softplus'
            % softplus (smoothed relu)
            res(i+1).x = max(zero, res(i).x) + log(1+exp(-abs(res(i).x)));
            
        case {'softmax', 'multinomial'},
            % softmax (multinomial)
            res(i+1).x = vl_nnsoftmax(res(i).x);
            
        case 'normalize',
            % contrast normalization
            res(i+1).x = vl_nnnormalize(res(i).x, l.param);
            
        case 'bernoulli_loss',
            % sigmoid
            res(i).x = one./(one+exp(-res(i).x));
            res(i).x = max(min(res(i).x, 1-1e-13), 1e-13); % for numerical stability
            
            % loss for binary input
            if ~isfield(l, 'data') || isempty(l.data),
                res(i+1).x = 0;
            else
                nsample = numel(res(i).x)/numel(l.data);
                res(i).x = reshape(res(i).x, [size(l.data), nsample]);
                res(i+1).x = -sum(sum(sum(bsxfun(@times, l.data, log(res(i).x)) + bsxfun(@times, 1-l.data, log(1-res(i).x)), 1), 2), 3);
            end
            
            % weights per sample
            if isfield(l, 'wps'),
                res(i+1).x = res(i+1).x(:)'*l.wps(:);
            else
                res(i+1).x = sum(res(i+1).x(:))/nsample;
            end
            cost = cost + res(i+1).x;
            
        case 'gaussian_loss',
            % loss for real input
            d = numel(res(i).x)/size(l.data, 4)/l.nsample;
            res(i).x = reshape(res(i).x, [size(l.data), l.nsample]);
            res(i).s = reshape(res(i).s, [size(l.data), l.nsample]);
            
            if ~isfield(l, 'data') || isempty(l.data),
                res(i+1).x = 0;
            else
                res(i+1).x = 0.5*d*log(2*pi);
                res(i+1).x = res(i+1).x + sum(sum(sum(log(res(i).s), 1), 2), 3);
                res(i+1).x = sum(sum(res(i+1).x + 0.5*sum(sum(sum(bsxfun(@rdivide, bsxfun(@minus, l.data, res(i).x).^2, res(i).s.^2), 1), 2), 3), 4), 5)/l.nsample;
            end
            cost = cost + res(i+1).x;
            
        case 'softmax_loss',
            % softmax
            hmax = max(res(i).x, [], 3);
            exph = exp(bsxfun(@minus, res(i).x, hmax));
            res(i).x = bsxfun(@rdivide, exph, sum(exph, 3));
            
            % loss for multinomial input
            if ~isfield(l, 'data') || isempty(l.data),
                res(i+1).x = 0;
            else
                nsample = numel(res(i).x)/numel(l.data);
                res(i).x = reshape(res(i).x, [size(l.data), nsample]);
                if isfield(l, 'w'),
                    % weighted average
                    res(i+1).x = -sum(sum(sum(bsxfun(@times, bsxfun(@times, l.data, l.w), log(res(i).x)), 3), 1), 2);
                else
                    % 1/n
                    res(i+1).x = -sum(sum(sum(bsxfun(@times, l.data, log(res(i).x)), 3), 1), 2);
                end
                
                % weights per sample
                if isfield(l, 'wps'),
                    res(i+1).x = res(i+1).x(:)'*l.wps(:);
                else
                    res(i+1).x = sum(res(i+1).x(:))/nsample;
                end
            end
            cost = cost + res(i+1).x;
            
        case 'kldiv_loss',
            % KL divergence
            res(i+1).x = res(i).x;
            res(i+1).s = res(i).s;
            cost = cost + 0.5*sum(sum(sum(sum(res(i).x.^2 + res(i).s.^2 - one - 2*log(res(i).s), 1), 2), 3), 4);
            
        case 'dropout'
            % dropout
            if opts.disableDropout
                res(i+1).x = res(i).x;
            elseif opts.freezeDropout
                [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux);
            else
                [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate);
            end
            
        otherwise
            error('Unknown layer type %s', l.type);
    end
    
    if gpuMode && opts.sync
        % This should make things slower, but on MATLAB 2014a it is necessary
        % for any decent performance.
        wait(gpuDevice);
    end
    
    res(i).time = toc(res(i).time);
    if opts.visualize,
        fprintf('layer [%d: %s]\t [%d x %d x %d x %d]\n', i, l.type, size(res(i).x, 1), size(res(i).x, 2), size(res(i).x, 3), size(res(i).x, 4));
    end
end

return;
