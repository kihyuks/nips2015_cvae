function net = mcn_cpu2gpu(net)

nlayer = length(net.layers);

for i = 1:nlayer,
    l = net.layers{i};
    
    if strcmp(l.type, 'conv') || strcmp(l.type, 'conv_valid') || strcmp(l.type, 'conv_full'),
        l.filters = gpuArray(l.filters);
        l.biases = gpuArray(l.biases);
        l.mgrad.filters = gpuArray(l.mgrad.filters);
        l.mgrad.biases = gpuArray(l.mgrad.biases);
        l.vgrad.filters = gpuArray(l.vgrad.filters);
        l.vgrad.biases = gpuArray(l.vgrad.biases);
    end
    
    if strcmp(l.type, 'conv_gaussian') || strcmp(l.type, 'conv_gaussian_valid') || strcmp(l.type, 'conv_gaussian_full'),
        l.filters = gpuArray(l.filters);
        l.biases = gpuArray(l.biases);
        l.mgrad.filters = gpuArray(l.mgrad.filters);
        l.mgrad.biases = gpuArray(l.mgrad.biases);
        l.vgrad.filters = gpuArray(l.vgrad.filters);
        l.vgrad.biases = gpuArray(l.vgrad.biases);
        l.filters_std = gpuArray(l.filters_std);
        l.biases_std = gpuArray(l.biases_std);
        l.mgrad.filters_std = gpuArray(l.mgrad.filters_std);
        l.mgrad.biases_std = gpuArray(l.mgrad.biases_std);
        l.vgrad.filters_std = gpuArray(l.vgrad.filters_std);
        l.vgrad.biases_std = gpuArray(l.vgrad.biases_std);
    end
    
    net.layers{i} = l;
end

return;