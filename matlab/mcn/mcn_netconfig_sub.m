% -----------------------------------------------------------------------
%   define network architecture
%   nh      : K x 1 (K+2 total layer including input and output)
%   nft       : (K+1) x 2
%   np     : K x 1
%   str      : (K+1) x 2
%   stotype     : K x 1, 0/1/2
%   ctype    : (K+1), cell
%   pltype      : 1, cell
%   rectype     : nonlinearity
%
% -----------------------------------------------------------------------

function [net, net_cls] = mcn_netconfig_sub(params)

if isfield(params, 'std_init'),
    std_init = params.std_init;
else
    std_init = 0.01;
end

nh = [params.numvis ; params.numhid(:)];
K = length(nh);
if ~isfield(params, 'numft'),
    nft = ones(K, 2);
else
    nft = params.numft;
end
if ~isfield(params, 'numpool'),
    np = ones(K-1, 1);
else
    np = params.numpool;
end
if ~isfield(params, 'stride'),
    str = ones(K, 1);
else
    str = params.stride;
end
if ~isfield(params, 'convtype'),
    ctype = cell(size(nft, 1), 1);
    for i = 1:length(ctype),
        ctype{i} = 'valid';
    end
else
    if iscell(params.convtype),
        ctype = params.convtype;
    else
        ctype = cell(size(nft, 1), 1);
        for i = 1:length(ctype),
            ctype{i} = params.convtype;
        end
    end
end
if ~isfield(params, 'pltype'),
    pltype = 'dense';
else
    pltype = params.pltype;
end
if ~isfield(params, 'stotype'),
    stotype = zeros(K-1, 1);
else
    stotype = params.stotype;
    if length(stotype) == 1,
        stotype = repmat(stotype, K-1, 1);
    end
end
if isfield(params, 'nsample'),
    nsample = params.nsample;
else
    nsample = 1;
end
if isfield(params, 'dropout'),
    dropout = params.dropout;
else
    dropout = 0;
end
if isfield(params, 'numlab'),
    numlab = params.numlab;
else
    numlab = params.numvis;
end

rectype = params.rectype;


% -----------------------------------------------------------------------
%                                                 recognition model
% -----------------------------------------------------------------------

rng('default');
net.layers = {};

% design network architecture
for i = 1:K-1,
    if stotype(i) == 2 || stotype(i) == 3,
        % do sampling
        if strcmp(ctype{i}, 'fconv_valid'),
            net.layers{end+1} = struct(...
                'type', 'conv_gaussian_valid', ...
                'filters', randn(1, 1, nft(i,1)*nft(i,2)*nh(i), nh(i+1), 'single')./sqrt(nft(i,1)*nft(i,2)*nh(i)), ...
                'biases', std_init * randn(1, nh(i+1),'single'), ...
                'fdim', [nft(i,1), nft(i,2), nh(i), nh(i+1)], ...
                'filters_std', zeros(1, 1, nft(i,1)*nft(i,2)*nh(i), nh(i+1), 'single')./sqrt(nft(i,1)*nft(i,2)*nh(i)), ...
                'biases_std', zeros(1, nh(i+1), 'single'), ...
                'share_std', 0, ...
                'stride', str(i), ...
                'pad', 0);
        elseif strcmp(ctype{i}, 'fconv_full'),
            net.layers{end+1} = struct(...
                'type', 'conv_gaussian_full', ...
                'filters', randn(1, 1, nh(i), nft(i,1)*nft(i,2)*nh(i+1), 'single')./sqrt(nft(i,1)*nft(i,2)*nh(i)), ...
                'biases', std_init * randn(1, nft(i,1)*nft(i,2)*nh(i+1),'single'), ...
                'fdim', [nft(i,1), nft(i,2), nh(i), nh(i+1)], ...
                'filters_std', zeros(1, 1, nh(i), nft(i,1)*nft(i,2)*nh(i+1), 'single')./sqrt(nft(i,1)*nft(i,2)*nh(i)), ...
                'biases_std', zeros(1, nft(i,1)*nft(i,2)*nh(i+1), 'single'), ...
                'share_std', 0, ...
                'stride', str(i), ...
                'pad', 0);
        else
            net.layers{end+1} = struct(...
                'type', 'conv_gaussian', ...
                'filters', randn(nft(i,1), nft(i,2), nh(i), nh(i+1), 'single')./sqrt(nft(i,1)*nft(i,2)*nh(i)), ...
                'biases', std_init * randn(1, nh(i+1),'single'), ...
                'filters_std', zeros(nft(i,1), nft(i,2), nh(i), nh(i+1), 'single')./sqrt(nft(i,1)*nft(i,2)*nh(i)), ...
                'biases_std', zeros(1, nh(i+1), 'single'), ...
                'share_std', 0, ...
                'stride', str(i), ...
                'pad', 0);
            if strcmp(ctype{i}, 'same'),
                net.layers{end}.pad = round([(nft(i,1)-1)/2, (nft(i,2)-1)/2, (nft(i,1)-1)/2, (nft(i,2)-1)/2]);
            elseif strcmp(ctype{i}, 'full'),
                net.layers{end}.pad = [nft(i,1)-1, nft(i,2)-1, nft(i,1)-1, nft(i,2)-1];
            end
        end
        
        if stotype(i) == 2,
            % gaussian sampling
            net.layers{end+1} = struct(...
                'type', 'gaussian_sampling', ...
                'nsample', nsample);
        elseif stotype(i) == 3,
            % KL-divergence loss
            net.layers{end+1} = struct('type', 'kldiv_loss');
            
            % gaussian sampling
            net.layers{end+1} = struct(...
                'type', 'gaussian_sampling', ...
                'nsample', nsample);
        end
    elseif stotype(i) == 0 || stotype(i) == 1,
        % no sampling
        if strcmp(ctype{i}, 'fconv_valid'),
            net.layers{end+1} = struct(...
                'type', 'conv_valid', ...
                'filters', randn(1, 1, nft(i,1)*nft(i,2)*nh(i), nh(i+1), 'single')./sqrt(nft(i,1)*nft(i,2)*nh(i)), ...
                'biases', std_init * randn(1, nh(i+1),'single'), ...
                'fdim', [nft(i,1), nft(i,2), nh(i), nh(i+1)], ...
                'stride', str(i), ...
                'pad', 0);
        elseif strcmp(ctype{i}, 'fconv_full'),
            net.layers{end+1} = struct(...
                'type', 'conv_full', ...
                'filters', randn(1, 1, nh(i), nft(i,1)*nft(i,2)*nh(i+1), 'single')./sqrt(nft(i,1)*nft(i,2)*nh(i)), ...
                'biases', std_init * randn(1, nft(i,1)*nft(i,2)*nh(i+1),'single'), ...
                'fdim', [nft(i,1), nft(i,2), nh(i), nh(i+1)], ...
                'stride', str(i), ...
                'pad', 0);
        else
            net.layers{end+1} = struct(...
                'type', 'conv', ...
                'filters', randn(nft(i,1), nft(i,2), nh(i), nh(i+1), 'single')./sqrt(nft(i,1)*nft(i,2)*nh(i)), ...
                'biases', std_init * randn(1, nh(i+1),'single'), ...
                'stride', str(i), ...
                'pad', 0);
            if strcmp(ctype{i}, 'same'),
                net.layers{end}.pad = round([(nft(i,1)-1)/2, (nft(i,2)-1)/2, (nft(i,1)-1)/2, (nft(i,2)-1)/2]);
            elseif strcmp(ctype{i}, 'full'),
                net.layers{end}.pad = [nft(i,1)-1, nft(i,2)-1, nft(i,1)-1, nft(i,2)-1];
            end
        end
        
        if stotype(i) == 0,
            % nonlinearity
            net.layers{end+1} = struct('type', rectype);
            
        elseif stotype(i) == 1,
            % KL-divergence loss
            net.layers{end+1} = struct('type', 'kldiv_loss');
        end
    end
    
    if np(i) > 1,
        net.layers{end+1} = struct(...
            'type', 'pool', ...
            'method', 'max', ...
            'pool', [np(i), np(i)], ...
            'stride', np(i), ...
            'pad', 0);
    elseif np(i) < -1,
        net.layers{end+1} = struct(...
            'type', 'unpool', ...
            'stride', abs(np(i)), ...
            'method', pltype);
    end
    if str(i) < -1,
        net.layers{end+1} = struct(...
            'type', 'unpool', ...
            'stride', str(i), ...
            'method', pltype);
    end
    if nft(i,1) == 1 && nft(i,2) == 1 && np(i) == 1 && dropout,
        net.layers{end+1} = struct(...
            'type', 'dropout', ...
            'rate', 0.5) ;
    end
end

% output layer
if isfield(params, 'typeout'),
    typeout = params.typeout;
else
    typeout = params.typein;
end

switch typeout,
    case {'bernoulli', 'binary'},
        net.layers{end+1} = struct(...
            'type', 'conv', ...
            'filters', randn(nft(K,1), nft(K,2), nh(end), numlab, 'single')./sqrt(nft(K,1)*nft(K,2)*nh(end)), ...
            'biases', std_init * randn(1, numlab, 'single'), ...
            'stride', str(K), ...
            'pad', 0);
        if strcmp(ctype{K}, 'same'),
            net.layers{end}.pad = round([(nft(K,1)-1)/2, (nft(K,2)-1)/2, (nft(K,1)-1)/2, (nft(K,2)-1)/2]);
        elseif strcmp(ctype{K}, 'full'),
            net.layers{end}.pad = [nft(K,1)-1, nft(K,2)-1, nft(K,1)-1, nft(K,2)-1];
        end
        
        % bernoulli loss
        net.layers{end+1} = struct('type', 'bernoulli_loss', 'nsample', nsample);
        
    case {'multinomial', 'softmax'},
        net.layers{end+1} = struct(...
            'type', 'conv', ...
            'filters', randn(nft(K,1), nft(K,2), nh(end), numlab, 'single')./sqrt(nft(K,1)*nft(K,2)*nh(end)), ...
            'biases', std_init * randn(1, numlab, 'single'), ...
            'stride', str(K), ...
            'pad', 0);
        if strcmp(ctype{K}, 'same'),
            net.layers{end}.pad = round([(nft(K,1)-1)/2, (nft(K,2)-1)/2, (nft(K,1)-1)/2, (nft(K,2)-1)/2]);
        elseif strcmp(ctype{K}, 'full'),
            net.layers{end}.pad = [nft(K,1)-1, nft(K,2)-1, nft(K,1)-1, nft(K,2)-1];
        end
        
        % softmax loss
        net.layers{end+1} = struct('type', 'softmax_loss', 'nsample', nsample);
        
    case {'real', 'gaussian'},
        net.layers{end+1} = struct(...
            'type', 'conv_gaussian', ...
            'filters', randn(nft(K,1), nft(K,2), nh(end), numlab, 'single')./sqrt(nft(K,1)*nft(K,2)*nh(end)), ...
            'biases', std_init * randn(1, numlab, 'single'), ...
            'filters_std', randn(nft(K,1), nft(K,2), nh(end), numlab, 'single')./sqrt(nft(K,1)*nft(K,2)*nh(end)), ...
            'biases_std', zeros(1, numlab, 'single'), ...
            'share_std', 0, ...
            'stride', str(K), ...
            'pad', 0);
        if strcmp(ctype{K}, 'same'),
            net.layers{end}.pad = round([(nft(K,1)-1)/2, (nft(K,2)-1)/2, (nft(K,1)-1)/2, (nft(K,2)-1)/2]);
        elseif strcmp(ctype{K}, 'full'),
            net.layers{end}.pad = [nft(K,1)-1, nft(K,2)-1, nft(K,1)-1, nft(K,2)-1];
        end
        
        % gaussian loss
        net.layers{end+1} = struct('type', 'gaussian_loss', 'nsample', nsample);
end

if nargout == 2,
    net_cls.layers = net.layers(end-1:end);
    net.layers = net.layers(1:end-2);
end

return;
