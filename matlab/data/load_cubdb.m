function [trainImgs, trainLabels, valImgs, valLabels, testImgs, testLabels] = load_cubdb(split, imsize)

if ~exist('imsize', 'var'),
    imsize = 128;
end
if ~exist('split', 'var'),
    split = 1;
end

startup;

datadir = [datadir 'CUB/'];
xtr_path = [datadir 'trainImgs.mat'];
ytr_path = [datadir 'trainLabels.mat'];
xts_path = [datadir 'testImgs.mat'];
yts_path = [datadir 'testLabels.mat'];

load(xtr_path, 'trainImgs');
load(ytr_path, 'trainLabels');
trainImgs = trainImgs./255;
trainLabels = floor(trainLabels ./ 255);

if imsize < 128,
    trainImgs_resize = zeros(imsize, imsize, 1, size(trainImgs, 4));
    trainLabels_resize = zeros(imsize, imsize, size(trainImgs, 4));
    trainImgs_resize = repmat(trainImgs_resize, [1 1 3 1]);
    for i = 1:size(trainImgs, 4),
        I = imresize(trainImgs(:, :, :, i), [imsize, imsize], 'bicubic');
        l = imresize(trainLabels(:, :, i), [imsize, imsize], 'nearest');
        trainImgs_resize(:, :, :, i) = I;
        trainLabels_resize(:, :, i) = l;
    end
    
    trainImgs = trainImgs_resize;
    trainLabels = trainLabels_resize;
    clear trainImgs_resize trainLabels_resize;
end

if split == 0,
    valImgs = [];
    valLabels = [];
else
    rng(split);
    idx = randperm(size(trainImgs, 4));
    validx = idx(1:round(size(trainImgs, 4)/10));
    tridx = idx(round(size(trainImgs, 4)/10)+1:end);
    
    valImgs = trainImgs(:, :, :, validx);
    valLabels = trainLabels(:, :, validx);
    trainImgs = trainImgs(:, :, :, tridx);
    trainLabels = trainLabels(:, :, tridx);
end

% horizontal flip on training set
trainImgs_flip = trainImgs(:,end:-1:1,:,:);
trainLabels_flip = trainLabels(:,end:-1:1,:);

trainImgs = cat(4, trainImgs, trainImgs_flip);
trainLabels = cat(3, trainLabels, trainLabels_flip);
clear trainImgs_flip trainLabels_flip;

if nargout >= 5,
    load(xts_path, 'testImgs');
    testImgs = testImgs./255;
    load(yts_path, 'testLabels');
    testLabels = floor(testLabels ./ 255);
    
    if imsize < 128,
        testImgs_resize = zeros(imsize, imsize, 1, size(testImgs, 4));
        testLabels_resize = zeros(imsize, imsize, size(testImgs, 4));
        testImgs_resize = repmat(testImgs_resize, [1 1 3 1]);
        for i = 1:size(testImgs, 4),
            I = imresize(testImgs(:, :, :, i), [imsize, imsize], 'bicubic');
            l = imresize(testLabels(:, :, i), [imsize, imsize], 'nearest');
            testImgs_resize(:, :, :, i) = I;
            testLabels_resize(:, :, i) = l;
        end
        
        testImgs = testImgs_resize;
        testLabels = testLabels_resize;
        clear testImgs_resize testLabels_resize;
    end
end

% [0, 1] -> [1, 2]
trainLabels = reshape(trainLabels + 1, imsize, imsize, 1, numel(trainLabels)/imsize^2);
valLabels = reshape(valLabels + 1, imsize, imsize, 1, numel(valLabels)/imsize^2);
testLabels = reshape(testLabels + 1, imsize, imsize, 1, numel(testLabels)/imsize^2);

return;