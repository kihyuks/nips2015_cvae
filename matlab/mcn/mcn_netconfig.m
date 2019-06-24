% -----------------------------------------------------------------------
%   define network architecture with multi-scale output training
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

function [net, net_cls] = mcn_netconfig(params, nscale)

if ~exist('nscale', 'var'),
    nscale = 1;
end
if nscale == 1,
    if nargout == 2,
        params.numft = [params.numft ; params.numft_cls];
        params.stride = [params.stride ; 1];
        net = cell(1, 1);
        net_cls = cell(1, 1);
        [net{1}, net_cls{1}] = mcn_netconfig_sub(params);
    else
        net = mcn_netconfig_sub(params);
    end
    return;
end

if isfield(params, 'std_init'),
    std_init = params.std_init;
else
    std_init = 0.01;
end

nh = params.numhid;
K = length(nh);
if ~isfield(params, 'numft'),
    nft = ones(K+1, 2);
else
    nft = params.numft;
end
if ~isfield(params, 'numpool'),
    np = ones(K, 1);
else
    np = params.numpool;
end
if ~isfield(params, 'stride'),
    str = ones(K+1, 1);
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
        ctype = cell(size(nft, 1)+1, 1);
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
    stotype = zeros(K, 1);
else
    stotype = params.stotype;
    if length(stotype) == 1,
        stotype = repmat(stotype, K, 1);
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

if ~isfield(params, 'numft_cls'),
    nft_cls = ones(length(nscale), 2);
else
    nft_cls = params.numft_cls;
end
if isfield(params, 'typeout'),
    typeout = params.typeout;
else
    typeout = params.typein;
end

numvis = params.numvis;
numlab = params.numlab;
rectype = params.rectype;


% -------------------------------------------------------------------------
%                                                       recognition model
% -------------------------------------------------------------------------

% vectorize
nh = nh(:);
np = np(:);
str = str(:);
stotype = stotype(:);

scale_change_idx = find(np < 0); % scale start to change once we get -1
scale_change_idx = scale_change_idx(end-nscale+1:end);

nh_c = cell(nscale, 1);
nft_c = cell(nscale, 1);
np_c = cell(nscale, 1);
str_c = cell(nscale, 1);
stotype_c = cell(nscale, 1);

nh_c{1} = nh(1:scale_change_idx(1));
nft_c{1} = [nft(1:scale_change_idx(1), :) ; nft_cls(1, :)];
np_c{1} = np(1:scale_change_idx(1));
str_c{1} = [str(1:scale_change_idx(1)) ; 1];
stotype_c{1} = stotype(1:scale_change_idx(1));
k = scale_change_idx(1);

for j = 2:nscale,
    numvis(j) = nh(k);
    nh_c{j} = nh(k+1:scale_change_idx(j));
    nft_c{j} = [nft(k+1:scale_change_idx(j), :) ; nft_cls(j, :)];
    np_c{j} = np(k+1:scale_change_idx(j));
    str_c{j} = [str(k+1:scale_change_idx(j)) ; 1];
    stotype_c{j} = stotype(k+1:scale_change_idx(j));
    k = scale_change_idx(j);
end

net = cell(nscale, 1);
net_cls = cell(nscale, 1);

for j = 1:nscale,
    params = struct(...
        'std_init', std_init, ...
        'numvis', numvis(j), ...
        'numlab', numlab, ...
        'typeout', typeout, ...
        'rectype', rectype, ...
        'nsample', nsample, ...
        'dropout', dropout, ...
        'pltype', pltype, ...
        'numhid', nh_c{j}, ...
        'numpool', np_c{j}, ...
        'numft', nft_c{j}, ...
        'stride', str_c{j}, ...
        'stotype', stotype_c{j});
    params.convtype = ctype;
    
    [net{j}, net_cls{j}] = mcn_netconfig_sub(params);
end

return;
