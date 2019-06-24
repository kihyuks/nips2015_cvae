function y_mult = multi_output(y,num_out,dim,outtype)
%%%
%   y       : 1       x batchsize
%   y_mult  : num_out x batchsize

if ~isa(y, 'double'),
    y = double(y);
end
if ~exist('outtype', 'var'),
    outtype = 'double';
end


if any(ismember(unique(y), 0)),
    y_mult = multi_output(y+1, num_out+1);
    y_mult = y_mult(2:end, :);
    
    ndim = size(y);
    y_mult = reshape(y_mult, [num_out, ndim([1:dim-1, dim+1:length(ndim)])]);
    
    ndim = ndims(y_mult);
    permidx = 1:ndim;
    permidx(1:dim-1) = permidx(2:dim);
    permidx(dim) = 1;
    y_mult = permute(y_mult, permidx);
else
    if ~exist('dim', 'var'),
        n = numel(y);
        y = y(:);
        
        y_mult = sparse(1:n,y,1,n,num_out,n); % numel x num_out
        y_mult = full(y_mult');
    else
        ndim = size(y);
        n = numel(y);
        y = y(:);
        
        y_mult = sparse(1:n,y,1,n,num_out,n); % numel x num_out
        y_mult = full(y_mult');
        y_mult = reshape(y_mult, [num_out, ndim([1:dim-1, dim+1:length(ndim)])]);
        
        ndim = ndims(y_mult);
        permidx = 1:ndim;
        permidx(1:dim-1) = permidx(2:dim);
        permidx(dim) = 1;
        y_mult = permute(y_mult, permidx);
    end
end

switch outtype,
    case 'single',
        y_mult = single(y_mult);
end

return;
