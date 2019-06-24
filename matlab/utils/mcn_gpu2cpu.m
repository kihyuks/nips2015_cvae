function net = mcn_gpu2cpu(net)

nlayer = length(net.layers);

for i = 1:nlayer,
    l = net.layers{i};
    
    if strcmp(l.type, 'conv') || strcmp(l.type, 'conv_valid') || strcmp(l.type, 'conv_full'),
        l.filters = gather(l.filters);
        l.biases = gather(l.biases);
        l.mgrad.filters = gather(l.mgrad.filters);
        l.mgrad.biases = gather(l.mgrad.biases);
        l.vgrad.filters = gather(l.vgrad.filters);
        l.vgrad.biases = gather(l.vgrad.biases);
    end
    
    if strcmp(l.type, 'conv_gaussian') || strcmp(l.type, 'conv_gaussian_valid') || strcmp(l.type, 'conv_gaussian_full'),
        l.filters = gather(l.filters);
        l.biases = gather(l.biases);
        l.mgrad.filters = gather(l.mgrad.filters);
        l.mgrad.biases = gather(l.mgrad.biases);
        l.vgrad.filters = gather(l.vgrad.filters);
        l.vgrad.biases = gather(l.vgrad.biases);
        l.filters_std = gather(l.filters_std);
        l.biases_std = gather(l.biases_std);
        l.mgrad.filters_std = gather(l.mgrad.filters_std);
        l.mgrad.biases_std = gather(l.mgrad.biases_std);
        l.vgrad.filters_std = gather(l.vgrad.filters_std);
        l.vgrad.biases_std = gather(l.vgrad.biases_std);
    end
    
    net.layers{i} = l;
end

return;