% -----------------------------------------------------------------------
%   Loading object segmentation dataset
%
%   - CUB (split 1-10), LFW (3way, split 1-5), LFW (4way, split 1-5)
%
%   - datasets are first introduced in the following papers:
%       [Max-Margin BMs for Object Segmentation, Yang et al., CVPR 2014]
%       [Augmenting CRFs with BM Shape Priors for Image Labeling,
%        Kae et al., CVPR 2013]
%       [What are good parts for hair shape modeling?,
%        Ai et al., CVPR 2012]
% -----------------------------------------------------------------------

function [xtr, ytr, xval, yval, xts, yts, numlab] = ...
    data_loader(dataset, split, imsize, do_mult)

if ~exist('split', 'var'),
    split = 1;
end
if ~exist('imsize', 'var'),
    imsize = 128;
end
if ~exist('do_mult', 'var'),
    do_mult = 1;
end

switch dataset,
    case 'cub',
        % split: 1-10
        [xtr, ytr, xval, yval, xts, yts] = load_cubdb(split, imsize);
        numlab = 2;
    case 'lfw',
        % split: 1-5
        [xtr, ytr, xval, yval, xts, yts, ~, numlab] = load_lfw(split, imsize);
end

if do_mult,
    ytr = single(multi_output(ytr, numlab, 3));
    yval = single(multi_output(yval, numlab, 3));
    yts = single(multi_output(yts, numlab, 3));
end

xtr = single(xtr);
xval = single(xval);
xts = single(xts);

return;