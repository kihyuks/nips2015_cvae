% -----------------------------------------------------------------------
%   make network (conditional variational auto-encoder)
%
%   params_xy2y     : (x,y) -> z -> y
%   params_x2z      : x -> z
%   params_x2y      : x -> y
%
% -----------------------------------------------------------------------

function [net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls] = ...
    mk_net(params_xy2y, params_x2z, params_x2y, nscale)

if ~exist('nscale', 'var'),
    nscale = 1;
end

% initialize xy to y
net_xy2y = mcn_netconfig(params_xy2y);
net_xy2y.layers = net_xy2y.layers(1:end-1);

% initialize x to z
net_x2z = mcn_netconfig(params_x2z);
net_x2z.layers = net_x2z.layers(1:end-3);

% initialize x to y
[net_x2y, net_cnn] = mcn_netconfig(params_x2y, nscale);

% decompose net_xy2y into net_xy2z and net_z2y
for i = 1:length(net_xy2y.layers),
    if strcmp(net_xy2y.layers{i}.type, 'kldiv_loss'),
        break;
    end
end

net_xy2z.layers = net_xy2y.layers(1:i-1);
net_xy2y.layers = net_xy2y.layers(i+1:end);
net_z2y = net_xy2y;
clear net_xy2y;

net_cls = cell(nscale, 1);
for j = 1:nscale,
    net_cls{j}.layers = net_cnn{j}.layers(end);
    net_cnn{j}.layers = net_cnn{j}.layers(1:end-1);
end

% initialize gradient and cast into gpu
net_xy2z = mcn_netinit(net_xy2z, params_xy2y);
net_z2y = mcn_netinit(net_z2y, params_xy2y);
net_x2z = mcn_netinit(net_x2z, params_x2z);
for j = 1:nscale,
    net_x2y{j} = mcn_netinit(net_x2y{j}, params_x2y);
    net_cnn{j} = mcn_netinit(net_cnn{j}, params_x2y);
end

return;