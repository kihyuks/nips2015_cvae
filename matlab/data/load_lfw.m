function [xtrain, ytrain, xval, yval, xtest, ytest, folds, numlab] = load_lfw(split, imsize)

startup;
datadir = [datadir '/LFW/'];

% load data directory
imgs = dir(datadir);
imgs(1:2) = [];
labels = imgs(2:2:end);
imgs = imgs(1:2:end);

ndata = length(imgs);

if ~exist('imsize', 'var'),
    imsize = 250;
end

x = zeros(imsize, imsize, 3, ndata);
y = zeros(imsize, imsize, 1, ndata);
for i = 1:ndata,
    % images
    I = imread([datadir '/' imgs(i).name]);
    if imsize < 250,
        I = imresize(I, [imsize, imsize], 'bicubic');
        x(:, :, :, i) = im2double(I);
    else
        x(:, :, :, i) = im2double(I);
    end
    
    % groundtruth seg. label
    I = imread([datadir '/' labels(i).name]);
    I = I./255;
    I = I(:,:,1) + I(:,:,2)*2 + I(:,:,3)*4;
    l = zeros(250, 250);
    l(I == 3) = 2; % face
    l(I == 4) = 3; % clothes
    l(I == 7) = 4; % hair
    l(I == 0) = 1; % background
    if imsize < 250,
        l = imresize(l, [imsize, imsize], 'nearest');
    end
    
    y(:, :, :, i) = l;
end


% make 5-fold split
nfolds = 5;
numlab = 4;
folds = 3*ones(ndata, nfolds);
for i = 1:nfolds,
    rng(i);
    idx = randperm(ndata);
    
    % 40% training(1), 10% validation(2), 50% testing(3)
    folds(idx(1:round(ndata/10)), i) = 2;
    folds(idx(round(ndata/10)+1:round(ndata/2)), i) = 1;
end


% horizontal flip
xflip = x(:,end:-1:1,:,:);
yflip = y(:,end:-1:1,:,:);

% use flipped version only for the training
folds_flip = folds;
folds_flip(folds_flip == 2) = 0;
folds_flip(folds_flip == 3) = 0;

x = cat(4, x, xflip);
y = cat(4, y, yflip);
folds = cat(1, folds, folds_flip);


if exist('split', 'var') && split <= nfolds && split > 0,
    xtrain = x(:,:,:,folds(:,split)==1);
    ytrain = y(:,:,:,folds(:,split)==1);
    xval = x(:,:,:,folds(:,split)==2);
    yval = y(:,:,:,folds(:,split)==2);
    xtest = x(:,:,:,folds(:,split)==3);
    ytest = y(:,:,:,folds(:,split)==3);
else
    xtrain = x;
    ytrain = y;
    xval = [];
    yval = [];
    xtest = [];
    ytest = [];
end

clear x y;

return;