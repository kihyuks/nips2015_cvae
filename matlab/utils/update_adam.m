
% -------------------------------------------------------------------------
%                               update parameters and accumulate gradient
%                               to minimize the objective function
% -------------------------------------------------------------------------

function [net, nandetected] = update_adam(net, res, lr, b1, b2, t)

eps = 1e-6; % for stability

% bias correction rate
if t < 10000,
    lr = lr*sqrt(1-(1-b2).^t)./(1-(1-b1).^t);
end

nlayer = length(net.layers);
nandetected = 0;

for i = 1:nlayer,
    l = net.layers{i};
    
    if strcmp(l.type, 'conv') || strcmp(l.type, 'conv_valid') || strcmp(l.type, 'conv_full'),
        grad = res(i);
        
        % update weights
        grad.filters = grad.filters + l.weightDecay*l.filters;
        grad.biases = grad.biases + l.weightDecay*l.biases;
        
        % biased first moment
        l.mgrad.filters = l.mgrad.filters + b1*(grad.filters - l.mgrad.filters);
        l.mgrad.biases = l.mgrad.biases + b1*(grad.biases - l.mgrad.biases);
        
        % biased second moment
        l.vgrad.filters = l.vgrad.filters + b2*(grad.filters.^2 - l.vgrad.filters);
        l.vgrad.biases = l.vgrad.biases + b2*(grad.biases.^2 - l.vgrad.biases);
        
        % update
        l.filters = l.filters - lr*l.learningRate*l.mgrad.filters./(sqrt(l.vgrad.filters) + eps);
        l.biases = l.biases - lr*l.learningRate*l.mgrad.biases./(sqrt(l.vgrad.biases) + eps);
        
        % check NaN
        if any(isnan(gather(l.filters(:)))),
            nandetected = 1;
            return;
        end
        
        if any(isnan(gather(l.biases(:)))),
            nandetected = 1;
            return;
        end
    end
    
    if strcmp(l.type, 'conv_gaussian') || strcmp(l.type, 'conv_gaussian_valid') || strcmp(l.type, 'conv_gaussian_full'),
        grad = res(i);
        
        % update weights
        grad.filters = grad.filters + l.weightDecay*l.filters;
        grad.biases = grad.biases + l.weightDecay*l.biases;
        
        % biased first moment
        l.mgrad.filters = l.mgrad.filters + b1*(grad.filters - l.mgrad.filters);
        l.mgrad.biases = l.mgrad.biases + b1*(grad.biases - l.mgrad.biases);
        
        % biased second moment
        l.vgrad.filters = l.vgrad.filters + b2*(grad.filters.^2 - l.vgrad.filters);
        l.vgrad.biases = l.vgrad.biases + b2*(grad.biases.^2 - l.vgrad.biases);
        
        % update
        l.filters = l.filters - lr*l.learningRate*l.mgrad.filters./(sqrt(l.vgrad.filters) + eps);
        l.biases = l.biases - lr*l.learningRate*l.mgrad.biases./(sqrt(l.vgrad.biases) + eps);
        
        % check NaN
        if any(isnan(gather(l.filters(:)))),
            nandetected = 1;
            return;
        end
        
        if any(isnan(gather(l.biases(:)))),
            nandetected = 1;
            return;
        end
        
        % update weights w.r.t. std
        grad.filters_std = grad.filters_std + l.weightDecay*l.filters_std;
        grad.biases_std = grad.biases_std + l.weightDecay*l.biases_std;
        
        % biased first moment
        l.mgrad.filters_std = l.mgrad.filters_std + b1*(grad.filters_std - l.mgrad.filters_std);
        l.mgrad.biases_std = l.mgrad.biases_std + b1*(grad.biases_std - l.mgrad.biases_std);
        
        % biased second moment
        l.vgrad.filters_std = l.vgrad.filters_std + b2*(grad.filters_std.^2 - l.vgrad.filters_std);
        l.vgrad.biases_std = l.vgrad.biases_std + b2*(grad.biases_std.^2 - l.vgrad.biases_std);
        
        % update
        l.filters_std = l.filters_std - lr*l.learningRate*l.mgrad.filters_std./(sqrt(l.vgrad.filters_std) + eps);
        l.biases_std = l.biases_std - lr*l.learningRate*l.mgrad.biases_std./(sqrt(l.vgrad.biases_std) + eps);
        
        % check NaN
        if any(isnan(gather(l.filters_std(:)))),
            nandetected = 1;
            return;
        end
        
        if any(isnan(gather(l.biases_std(:)))),
            nandetected = 1;
            return;
        end
    end
    
    net.layers{i} = l;
end

return;


