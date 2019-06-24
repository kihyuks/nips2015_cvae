function sample = mcn_sample(net, optgpu, imsize, batchsize, res)

startup;
startup_matconvnet;

if ~exist('optgpu', 'var'),
    optgpu = 1;
end
if ~exist('batchsize', 'var'),
    batchsize = 100;
end
if ~exist('res', 'var'),
    res = [];
end


if optgpu,
    net = mcn_cpu2gpu(net);
else
    net = mcn_gpu2cpu(net);
end

n = length(net.layers);

% find gaussian sampling layer
for i = 1:n,
    if strcmp(net.layers{i}.type, 'gaussian_sampling'),
        break;
    end
end
x = randn([imsize, batchsize], 'single');
if optgpu,
    x = gpuArray(x);
end
res_tmp = mcn_ff(net, x, [], [], struct('evalLayer', i));
x = zeros(size(res_tmp(end-1).x), 'single');
s = ones(size(res_tmp(end-1).s), 'single');
if optgpu,
    x = gpuArray(x);
    s = gpuArray(s);
end


% create net from sampling
net_sample.layers = net.layers(i:end);

if strcmp(net_sample.layers{end}.type, 'bernoulli_loss'),
    net_sample.layers{end}.type = 'bernoulli';
elseif strcmp(net_sample.layers{end}.type, 'softmax_loss'),
    net_sample.layers{end}.type = 'softmax';
elseif strcmp(net_sample.layers{end}.type, 'gaussian_loss'),
    net_sample.layers{end}.type = 'gaussian';
elseif strcmp(net_sample.layers{end}.type, 'gaussianbb_loss'),
    net_sample.layers{end}.type = 'gaussian';
    net_sample.layers{end+1}.type = 'sigmoid';    
end
clear net;

if optgpu,
    net_sample = mcn_cpu2gpu(net_sample);
else
    net_sample = mcn_gpu2cpu(net_sample);
end

[res, ~] = mcn_ff(net_sample, x, s, res);
sample = gather(res(end).x);

return;