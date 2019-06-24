% =====================================
% Generate sample from distribution x
% =====================================
% x     : numlab x batchsize

function [y, x] = realize_mult(x, dim)

nd = ndims(x);
sz = size(x);
nl = size(x, dim);

if isa(x, 'single'),
    dosingle = 1;
else
    dosingle = 0;
end

% permute and reshape
if dim ~= 1,
    x = permute(x, [dim, setdiff(1:nd, dim)]);
end
if nd > 2,
    x = reshape(x, nl, numel(x)/nl);
end

x = double(x)';
x = bsxfun(@rdivide, x, sum(x, 2));

cumx = cumsum(x, 2);
unifrnd = rand(size(x, 1), 1);
temp = bsxfun(@gt, cumx, unifrnd);
yidx = diff(temp, 1, 2);
y = zeros(size(x));
y(:,1) = 1-sum(yidx,2);
y(:,2:end) = yidx;

x = x';
y = y';


% reshape back to original dimension
y = reshape(y, [nl, sz(setdiff(1:nd, dim))]);
y = permute(y, [2:dim, 1, dim+1:nd]);
if dosingle,
    y = single(y);
end

if nargout == 2,
    x = reshape(x, [nl, sz(setdiff(1:nd, dim))]);
    x = permute(x, [2:dim, 1, dim+1:nd]);
    if dosingle,
        x = single(x);
    end
end

return;
