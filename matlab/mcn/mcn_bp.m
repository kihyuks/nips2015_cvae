% -----------------------------------------------------------------------
%   matConvNet -- reformatted to support two input variables, x and s,
%   (backward)    for gaussian random variable.
%
% -----------------------------------------------------------------------

function res = mcn_bp(net, dzdy, res, varargin)


% -----------------------------------------------------------------------
%                                                  default parameters
% -----------------------------------------------------------------------

opts.res = [];
opts.conserveMemory = false;
opts.sync = false;
opts.disableDropout = false;
opts.freezeDropout = false;
opts.gradcheck = false;
opts.bpLayer_init = numel(net.layers);
opts.bpLayer_end = 1;

opts = vl_argparse(opts, varargin);


% -----------------------------------------------------------------------
%                                                      initialization
% -----------------------------------------------------------------------

nlayer = numel(net.layers);
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

if isempty(dzdy),
    gpuMode = isa(res(end).dzdx, 'gpuArray');
else
    gpuMode = isa(dzdy, 'gpuArray');
    res(nlayer+1).dzdx = dzdy;
end

if opts.gradcheck,
    one = double(1);
    zero = double(0);
else
    one = single(1);
    zero = single(0);
end
if gpuMode,
    one = gpuArray(one);
    zero = gpuArray(zero);
end


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

for i = opts.bpLayer_init:-1:opts.bpLayer_end,
    l = net.layers{i};
    res(i).backwardTime = tic;
    
    switch l.type
        case 'conv',
            if opts.gradcheck,
                if length(l.pad) > 1 && sum(l.pad) > 0,
                    % linear (conv) response for mean
                    dzdx = zeros(size(res(i+1).dzdx,1)+size(l.filters,1)-1, size(res(i+1).dzdx,2)+size(l.filters,2)-1, size(l.filters,3), size(res(i+1).dzdx,4));
                    for nh = 1:size(l.filters, 3),
                        for nv = 1:size(res(i+1).dzdx, 3),
                            dzdx(:, :, nh, :) = dzdx(:, :, nh, :) + convn(res(i+1).dzdx(:, :, nv, :), l.filters(:, :, nh, nv), 'full');
                        end
                    end
                    res(i).dzdx = dzdx(l.pad(1)+1:size(dzdx,1)-l.pad(3), l.pad(2)+1:size(dzdx,2)-l.pad(4), :, :);
                    
                    x = zeros(size(res(i).x,1) + l.pad(1) + l.pad(3), size(res(i).x,2) + l.pad(2) + l.pad(4), size(res(i).x,3), size(res(i).x,4));
                    x(l.pad(1)+1:l.pad(1)+size(res(i).x,1), l.pad(2)+1:l.pad(2)+size(res(i).x,2), :, :) = res(i).x;
                    
                    res(i).filters = zeros(size(l.filters));
                    for nv = 1:size(l.filters, 3),
                        for nh = 1:size(l.filters, 4),
                            res(i).filters(:, :, nv, nh) = convn(x(:, :, nv, :), res(i+1).dzdx(end:-1:1, end:-1:1, nh, end:-1:1), 'valid');
                        end
                    end
                    
                    res(i).biases = zeros(1, size(l.filters, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i).biases(1, nh) = sum(sum(sum(res(i+1).dzdx(:, :, nh, :), 4), 1), 2);
                    end
                else
                    % linear (conv) response for mean
                    res(i).dzdx = zeros(size(res(i+1).dzdx,1)+size(l.filters,1)-1, size(res(i+1).dzdx,2)+size(l.filters,2)-1, size(l.filters,3), size(res(i+1).dzdx,4));
                    for nh = 1:size(l.filters, 3),
                        for nv = 1:size(res(i+1).dzdx, 3),
                            res(i).dzdx(:, :, nh, :) = res(i).dzdx(:, :, nh, :) + convn(res(i+1).dzdx(:, :, nv, :), l.filters(:, :, nh, nv), 'full');
                        end
                    end
                    
                    res(i).filters = zeros(size(l.filters));
                    for nv = 1:size(l.filters, 3),
                        for nh = 1:size(l.filters, 4),
                            res(i).filters(:, :, nv, nh) = convn(res(i).x(:, :, nv, :), res(i+1).dzdx(end:-1:1, end:-1:1, nh, end:-1:1), 'valid');
                        end
                    end
                    
                    res(i).biases = zeros(1, size(l.filters, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i).biases(1, nh) = sum(sum(sum(res(i+1).dzdx(:, :, nh, :), 4), 1), 2);
                    end
                end
            else
                % linear (convolutional) response
                [res(i).dzdx, res(i).filters, res(i).biases] = ...
                    vl_nnconv(res(i).x, l.filters, l.biases, ...
                    res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
            end
            
        case 'conv_valid',
            % linear (convolutional) response
            % convolution
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
            
            % reshape
            dimx = size(res(i).dzdx);
            dimf = l.fdim;
            res(i).dzdx = reshape(res(i).dzdx, dimf(1), dimf(2), dimx(3)./(dimf(1)*dimf(2)), dimx(4));
            
        case 'conv_full',
            % linear (convolutional) response
            % reshape
            dimx = size(res(i+1).dzdx);
            dimf = l.fdim;
            res(i+1).dzdx = reshape(res(i+1).dzdx, dimx(1)./dimf(1), dimx(2)./dimf(2), dimf(1)*dimf(2)*dimx(3), dimx(4));
            
            % convolution
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
            
        case 'conv_gaussian',
            if ~isfield(l, 'share_std'),
                l.share_std = 0;
            end
            
            if l.share_std == 1, % spatial
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, 1]);
            elseif l.share_std == 2, % spatial + channel
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, size(l.filters, 4)]);
            end
            
            if opts.gradcheck,
                if length(l.pad) > 1 && sum(l.pad) > 0,
                    % linear (conv) response for mean
                    dzdx = zeros(size(res(i+1).dzdx,1)+size(l.filters,1)-1, size(res(i+1).dzdx,2)+size(l.filters,2)-1, size(l.filters,3), size(res(i+1).dzdx,4));
                    for nh = 1:size(l.filters, 3),
                        for nv = 1:size(res(i+1).dzdx, 3),
                            dzdx(:, :, nh, :) = dzdx(:, :, nh, :) + convn(res(i+1).dzdx(:, :, nv, :), l.filters(:, :, nh, nv), 'full');
                        end
                    end
                    res(i).dzdx = dzdx(l.pad(1)+1:size(dzdx,1)-l.pad(3), l.pad(2)+1:size(dzdx,2)-l.pad(4), :, :);
                    
                    x = zeros(size(res(i).x,1) + l.pad(1) + l.pad(3), size(res(i).x,2) + l.pad(2) + l.pad(4), size(res(i).x,3), size(res(i).x,4));
                    x(l.pad(1)+1:l.pad(1)+size(res(i).x,1), l.pad(2)+1:l.pad(2)+size(res(i).x,2), :, :) = res(i).x;
                    
                    res(i).filters = zeros(size(l.filters));
                    for nv = 1:size(l.filters, 3),
                        for nh = 1:size(l.filters, 4),
                            res(i).filters(:, :, nv, nh) = convn(x(:, :, nv, :), res(i+1).dzdx(end:-1:1, end:-1:1, nh, end:-1:1), 'valid');
                        end
                    end
                    
                    res(i).biases = zeros(1, size(l.filters, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i).biases(1, nh) = sum(sum(sum(res(i+1).dzdx(:, :, nh, :), 4), 1), 2);
                    end
                    
                    % linear (conv) response for std
                    dzds = zeros(size(res(i+1).dzds,1)+size(l.filters_std,1)-1, size(res(i+1).dzds,2)+size(l.filters_std,2)-1, size(l.filters_std,3), size(res(i+1).dzds,4));
                    for nh = 1:size(l.filters_std, 3),
                        for nv = 1:size(res(i+1).dzds, 3),
                            dzds(:, :, nh, :) = dzds(:, :, nh, :) + convn(res(i+1).dzds(:, :, nv, :), l.filters_std(:, :, nh, nv), 'full');
                        end
                    end
                    res(i).dzds = dzds(l.pad(1)+1:size(dzds,1)-l.pad(3), l.pad(2)+1:size(dzds,2)-l.pad(4), :, :);
                    
                    x = zeros(size(res(i).x,1) + l.pad(1) + l.pad(3), size(res(i).x,2) + l.pad(2) + l.pad(4), size(res(i).x,3), size(res(i).x,4));
                    x(l.pad(1)+1:l.pad(1)+size(res(i).x,1), l.pad(2)+1:l.pad(2)+size(res(i).x,2), :, :) = res(i).x;
                    
                    res(i).filters_std = zeros(size(l.filters_std));
                    for nv = 1:size(l.filters_std, 3),
                        for nh = 1:size(l.filters_std, 4),
                            res(i).filters_std(:, :, nv, nh) = convn(x(:, :, nv, :), res(i+1).dzds(end:-1:1, end:-1:1, nh, end:-1:1), 'valid');
                        end
                    end
                    
                    res(i).biases_std = zeros(1, size(l.filters_std, 4));
                    for nh = 1:size(l.filters_std, 4),
                        res(i).biases_std(1, nh) = sum(sum(sum(res(i+1).dzds(:, :, nh, :), 4), 1), 2);
                    end
                else
                    % linear (conv) response for mean
                    res(i).dzdx = zeros(size(res(i+1).dzdx,1)+size(l.filters,1)-1, size(res(i+1).dzdx,2)+size(l.filters,2)-1, size(l.filters,3), size(res(i+1).dzdx,4));
                    for nh = 1:size(l.filters, 3),
                        for nv = 1:size(res(i+1).dzdx, 3),
                            res(i).dzdx(:, :, nh, :) = res(i).dzdx(:, :, nh, :) + convn(res(i+1).dzdx(:, :, nv, :), l.filters(:, :, nh, nv), 'full');
                        end
                    end
                    
                    res(i).filters = zeros(size(l.filters));
                    for nv = 1:size(l.filters, 3),
                        for nh = 1:size(l.filters, 4),
                            res(i).filters(:, :, nv, nh) = convn(res(i).x(:, :, nv, :), res(i+1).dzdx(end:-1:1, end:-1:1, nh, end:-1:1), 'valid');
                        end
                    end
                    
                    res(i).biases = zeros(1, size(l.filters, 4));
                    for nh = 1:size(l.filters, 4),
                        res(i).biases(1, nh) = sum(sum(sum(res(i+1).dzdx(:, :, nh, :), 4), 1), 2);
                    end
                    
                    % linear (conv) response for std
                    res(i).dzds = zeros(size(res(i+1).dzds,1)+size(l.filters_std,1)-1, size(res(i+1).dzds,2)+size(l.filters_std,2)-1, size(l.filters_std,3), size(res(i+1).dzds,4));
                    for nh = 1:size(l.filters_std, 3),
                        for nv = 1:size(res(i+1).dzds, 3),
                            res(i).dzds(:, :, nh, :) = res(i).dzds(:, :, nh, :) + convn(res(i+1).dzds(:, :, nv, :), l.filters_std(:, :, nh, nv), 'full');
                        end
                    end
                    
                    res(i).filters_std = zeros(size(l.filters_std));
                    for nv = 1:size(l.filters_std, 3),
                        for nh = 1:size(l.filters_std, 4),
                            res(i).filters_std(:, :, nv, nh) = convn(res(i).x(:, :, nv, :), res(i+1).dzds(end:-1:1, end:-1:1, nh, end:-1:1), 'valid');
                        end
                    end
                    
                    res(i).biases_std = zeros(1, size(l.filters_std, 4));
                    for nh = 1:size(l.filters_std, 4),
                        res(i).biases_std(1, nh) = sum(sum(sum(res(i+1).dzds(:, :, nh, :), 4), 1), 2);
                    end
                end
            else
                % linear (conv) response for mean
                [res(i).dzdx, res(i).filters, res(i).biases] = ...
                    vl_nnconv(res(i).x, l.filters, l.biases, ...
                    res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
                
                % linear (conv) response for std
                [res(i).dzds, res(i).filters_std, res(i).biases_std] = ...
                    vl_nnconv(res(i).x, l.filters_std, l.biases_std, ...
                    res(i+1).dzds, 'pad', l.pad, 'stride', l.stride);
            end
            
            if l.share_std == 1,
                res(i).filters_std = sum(sum(res(i).filters_std, 1), 2);
            elseif l.share_std == 2,
                res(i).filters_std = sum(sum(sum(res(i).filters_std, 1), 2), 4);
            end
            res(i).dzdx = res(i).dzdx + res(i).dzds;
            
        case 'conv_gaussian_valid',
            if ~isfield(l, 'share_std'),
                l.share_std = 0;
            end
            
            if l.share_std == 1, % spatial
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, 1]);
            elseif l.share_std == 2, % spatial + channel
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, size(l.filters, 4)]);
            end
            
            % linear (conv) response for mean
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
            
            % linear (conv) response for std
            [res(i).dzds, res(i).filters_std, res(i).biases_std] = ...
                vl_nnconv(res(i).x, l.filters_std, l.biases_std, ...
                res(i+1).dzds, 'pad', l.pad, 'stride', l.stride);
            
            if l.share_std == 1,
                res(i).filters_std = sum(sum(res(i).filters_std, 1), 2);
            elseif l.share_std == 2,
                res(i).filters_std = sum(sum(sum(res(i).filters_std, 1), 2), 4);
            end
            
            % reshape
            dimx = size(res(i).dzdx);
            dimf = l.fdim;
            res(i).dzdx = reshape(res(i).dzdx, dimf(1), dimf(2), dimx(3)./(dimf(1)*dimf(2)), dimx(4));
            res(i).dzds = reshape(res(i).dzds, dimf(1), dimf(2), dimx(3)./(dimf(1)*dimf(2)), dimx(4));
            
            res(i).dzdx = res(i).dzdx + res(i).dzds;
            
        case 'conv_gaussian_full',
            if ~isfield(l, 'share_std'),
                l.share_std = 0;
            end
            
            if l.share_std == 1, % spatial
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, 1]);
            elseif l.share_std == 2, % spatial + channel
                l.filters_std = repmat(l.filters_std, [size(l.filters, 1), size(l.filters, 2), 1, size(l.filters, 4)]);
            end
            
            % reshape
            dimx = size(res(i+1).dzdx);
            dimf = l.fdim;
            res(i+1).dzdx = reshape(res(i+1).dzdx, dimx(1)./dimf(1), dimx(2)./dimf(2), dimf(1)*dimf(2)*dimx(3), dimx(4));
            res(i+1).dzds = reshape(res(i+1).dzds, dimx(1)./dimf(1), dimx(2)./dimf(2), dimf(1)*dimf(2)*dimx(3), dimx(4));
            
            % linear (conv) response for mean
            [res(i).dzdx, res(i).filters, res(i).biases] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride);
            
            % linear (conv) response for std
            [res(i).dzds, res(i).filters_std, res(i).biases_std] = ...
                vl_nnconv(res(i).x, l.filters_std, l.biases_std, ...
                res(i+1).dzds, 'pad', l.pad, 'stride', l.stride);
            
            if l.share_std == 1,
                res(i).filters_std = sum(sum(res(i).filters_std, 1), 2);
            elseif l.share_std == 2,
                res(i).filters_std = sum(sum(sum(res(i).filters_std, 1), 2), 4);
            end
            res(i).dzdx = res(i).dzdx + res(i).dzds;
            
        case 'gaussian_sampling',
            % gaussian sampling
            res(i+1).dzdx = reshape(res(i+1).dzdx, [size(res(i).s), l.nsample]);
            res(i).dzds = 0.5*sum(res(i+1).dzdx.*res(i+1).n, 5).*res(i).s;
            res(i).dzdx = sum(res(i+1).dzdx, 5);
            
        case 'pool',
            % pooling
            res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                'pad', l.pad, 'stride', l.stride, 'method', l.method);
            
        case 'unpool',
            % unpooling
            res(i).dzdx = vl_nnupsample(res(i).x, res(i+1).dzdx, 'stride', l.stride, 'method', l.method);
            
        case {'gaussian', 'linear'}
            % linear
            res(i).dzdx = res(i+1).dzdx;
            res(i).dzds = res(i+1).dzds;
            
        case {'sigmoid', 'bernoulli'},
            % sigmoid
            res(i+1).x = reshape(res(i+1).x, size(res(i+1).dzdx));
            res(i).dzdx = res(i+1).x.*(one-res(i+1).x).*res(i+1).dzdx;
            
        case 'relu',
            % rectified linear
            res(i).dzdx = (res(i).x > zero).*res(i+1).dzdx;
            
        case 'softplus',
            % softplus (smoothed relu)
            res(i).dzdx = ((exp(res(i+1).x)-one)./exp(res(i+1).x)).*res(i+1).dzdx;
            
        case {'softmax', 'multinomial'},
            % softmax (multinomial)
            res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx);
            
        case 'normalize',
            % contrast normalization
            res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx);
            
        case 'bernoulli_loss',
            nsample = numel(res(i).x)/numel(l.data);
            
            if nsample == 1,
                l.data = reshape(l.data, size(res(i).x));
            else
                res(i).x = reshape(res(i).x, [size(l.data), nsample]);
            end
            
            % loss for binary input + sigmoid
            if isfield(l, 'wps'),
                wps = reshape(l.wps, [1, 1, 1, size(res(i).x, 4), size(res(i).x, 5)]);
                res(i).dzdx = res(i+1).dzdx*bsxfun(@times, bsxfun(@minus, res(i).x, l.data), wps);
                res(i).dzdx = reshape(res(i).dzdx, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*nsample);
            else
                res(i).dzdx = res(i+1).dzdx*bsxfun(@minus, res(i).x, l.data);
                res(i).dzdx = reshape(res(i).dzdx, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*nsample)/nsample;
            end
            
        case 'gaussian_loss',
            if l.nsample == 1,
                l.data = reshape(l.data, size(res(i).x));
            else
                res(i).x = reshape(res(i).x, [size(l.data), l.nsample]);
            end
            
            % loss for real input
            res(i).dzdx = res(i+1).dzdx*bsxfun(@rdivide, bsxfun(@minus, res(i).x, l.data), res(i).s.^2);
            res(i).dzdx = reshape(res(i).dzdx, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*l.nsample)/l.nsample;
            
            res(i).dzds = 0.5*(one - bsxfun(@rdivide, bsxfun(@minus, res(i).x, l.data), res(i).s).^2);
            res(i).dzds = reshape(res(i).dzds, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*l.nsample)/l.nsample;
            
        case 'softmax_loss',
            nsample = numel(res(i).x)/numel(l.data);
            
            if nsample == 1,
                l.data = reshape(l.data, size(res(i).x));
            else
                res(i).x = reshape(res(i).x, [size(l.data), nsample]);
            end
            
            % loss for multinomial input
            if isfield(l, 'w'),
                tmp = bsxfun(@times, l.data, l.w);
                res(i).dzdx = res(i+1).dzdx*bsxfun(@minus, bsxfun(@times, res(i).x, sum(tmp, 3)), tmp);
            else
                res(i).dzdx = res(i+1).dzdx*bsxfun(@minus, res(i).x, l.data);
            end
            
            if isfield(l, 'wps'),
                wps = reshape(l.wps, [1, 1, 1, size(res(i).x, 4), size(res(i).x, 5)]);
                res(i).dzdx = bsxfun(@times, bsxfun(@minus, res(i).x, l.data), wps);
                res(i).dzdx = reshape(res(i).dzdx, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*nsample);
            else
                res(i).dzdx = bsxfun(@times, res(i).dzdx, sum(l.data, 3));
                res(i).dzdx = reshape(res(i).dzdx, size(l.data, 1), size(l.data, 2), size(l.data, 3), size(l.data, 4)*nsample)/nsample;
            end
            
        case 'kldiv_loss',
            % KL divergence
            res(i).dzdx = res(i+1).dzdx + res(i).x;
            res(i).dzds = res(i+1).dzds + 0.5*(res(i).s.^2 - one);
            
        case 'dropout'
            if opts.disableDropout
                res(i).dzdx = res(i+1).dzdx;
            else
                res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, 'mask', res(i+1).aux);
            end
            
        otherwise
            error('Unknown layer type %s', l.type);
    end
    
    if gpuMode && opts.sync,
        % This should make things slower, but on MATLAB 2014a it is necessary
        % for any decent performance.
        wait(gpuDevice);
    end
    
    res(i).backwardTime = toc(res(i).backwardTime);
end


return;
