% -----------------------------------------------------------------------
%                      network initialization for training with AdaM
% -----------------------------------------------------------------------

function net = mcn_netinit(net, params)

if exist('params', 'var'),
    if ~isfield(params, 'optgpu'),
        optgpu = 0;
    else
        optgpu = params.optgpu;
    end
    if ~isfield(params, 'l2reg'),
        l2reg = 0;
    else
        l2reg = params.l2reg;
    end
else
    optgpu = 0;
    l2reg = 0;
end


% -----------------------------------------------------------------------
%                                                   initialize model
% -----------------------------------------------------------------------

nlayer = length(net.layers);

for i = 1:nlayer,
    l = net.layers{i};
    
    if strcmp(l.type, 'conv') || strcmp(l.type, 'conv_valid') || strcmp(l.type, 'conv_full'),
        l.mgrad.filters = zeros(size(l.filters), 'single');
        l.mgrad.biases = zeros(size(l.biases), 'single');
        l.vgrad.filters = zeros(size(l.filters), 'single');
        l.vgrad.biases = zeros(size(l.biases), 'single');
        if optgpu,
            l.filters = gpuArray(l.filters);
            l.biases = gpuArray(l.biases);
            l.mgrad.filters = gpuArray(l.mgrad.filters);
            l.mgrad.biases = gpuArray(l.mgrad.biases);
            l.vgrad.filters = gpuArray(l.vgrad.filters);
            l.vgrad.biases = gpuArray(l.vgrad.biases);
        end
        
        if ~isfield(l, 'learningRate'),
            l.learningRate = 1;
        end
        if ~isfield(l, 'weightDecay'),
            l.weightDecay = l2reg;
        end
    end
    
    if strcmp(l.type, 'conv_gaussian') || strcmp(l.type, 'conv_gaussian_valid') || strcmp(l.type, 'conv_gaussian_full'),
        l.mgrad.filters = zeros(size(l.filters), 'single');
        l.mgrad.biases = zeros(size(l.biases), 'single');
        l.vgrad.filters = zeros(size(l.filters), 'single');
        l.vgrad.biases = zeros(size(l.biases), 'single');
        if optgpu,
            l.filters = gpuArray(l.filters);
            l.biases = gpuArray(l.biases);
            l.mgrad.filters = gpuArray(l.mgrad.filters);
            l.mgrad.biases = gpuArray(l.mgrad.biases);
            l.vgrad.filters = gpuArray(l.vgrad.filters);
            l.vgrad.biases = gpuArray(l.vgrad.biases);
        end
        
        l.mgrad.filters_std = zeros(size(l.filters_std), 'single');
        l.mgrad.biases_std = zeros(size(l.biases_std), 'single');
        l.vgrad.filters_std = zeros(size(l.filters_std), 'single');
        l.vgrad.biases_std = zeros(size(l.biases_std), 'single');
        if optgpu,
            l.filters_std = gpuArray(l.filters_std);
            l.biases_std = gpuArray(l.biases_std);
            l.mgrad.filters_std = gpuArray(l.mgrad.filters_std);
            l.mgrad.biases_std = gpuArray(l.mgrad.biases_std);
            l.vgrad.filters_std = gpuArray(l.vgrad.filters_std);
            l.vgrad.biases_std = gpuArray(l.vgrad.biases_std);
        end
        
        if ~isfield(l, 'learningRate'),
            l.learningRate = 1;
        end
        if ~isfield(l, 'weightDecay'),
            l.weightDecay = l2reg;
        end
    end
    net.layers{i} = l;
end


return;
