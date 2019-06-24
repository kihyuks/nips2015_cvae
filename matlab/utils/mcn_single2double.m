function net = mcn_single2double(net)

nlayer = length(net.layers);

for i = 1:nlayer,
    l = net.layers{i};
    
    if strcmp(l.type, 'conv'),
        l.filters = double(gather(l.filters));
        l.biases = double(gather(l.biases));
        l.mgrad.filters = double(gather(l.mgrad.filters));
        l.mgrad.biases = double(gather(l.mgrad.biases));
        l.vgrad.filters = double(gather(l.vgrad.filters));
        l.vgrad.biases = double(gather(l.vgrad.biases));
    end
    
    if strcmp(l.type, 'conv_gaussian'),
        l.filters = double(gather(l.filters));
        l.biases = double(gather(l.biases));
        l.mgrad.filters = double(gather(l.mgrad.filters));
        l.mgrad.biases = double(gather(l.mgrad.biases));
        l.vgrad.filters = double(gather(l.vgrad.filters));
        l.vgrad.biases = double(gather(l.vgrad.biases));
        l.filters_std = double(gather(l.filters_std));
        l.biases_std = double(gather(l.biases_std));
        l.mgrad.filters_std = double(gather(l.mgrad.filters_std));
        l.mgrad.biases_std = double(gather(l.mgrad.biases_std));
        l.vgrad.filters_std = double(gather(l.vgrad.filters_std));
        l.vgrad.biases_std = double(gather(l.vgrad.biases_std));
    end
    
    net.layers{i} = l;
end

return;