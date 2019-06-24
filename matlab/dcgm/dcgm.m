% -----------------------------------------------------------------------
%   conditional variational auto-encoder:
%       -KL(q(z|x,y) || p(z|x)) + log p(y|x,z), z ~ q(z|x,y)
%
%   variation:
%       1. recurrent encoding
%       2. multi-scale (defined by numft_cls)
%       3. noise-injection training
%
%   written by Kihyuk Sohn
% -----------------------------------------------------------------------

function [net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params, t, fname] = ...
    dcgm(dataset, optgpu, dotest, split, imsize, do_recurrent, ...
    droprate, nettype, numft_cls, nsample, alpha, lr, lr_decay, l2reg, ...
    dropout, maxiter, saveiter, batchsize, dosample, std_init)

startup;
startup_matconvnet;

if ~exist('dataset', 'var'),
    dataset = 'cub'; end
if ~exist('optgpu', 'var'),
    optgpu = 1; end
if ~exist('dotest', 'var'),
    dotest = 'acc'; end % 'acc', 'll'
if ~exist('split', 'var'),
    split = 1; end
if ~exist('imsize', 'var'),
    imsize = 128; end
if ~exist('do_recurrent', 'var'),
    do_recurrent = 1; end
if ~exist('droprate', 'var'),
    droprate = 0; end
if ~exist('nsample', 'var'),
    nsample = 1; end
if ~exist('alpha', 'var'),
    alpha = 0.5; end
if ~exist('lr', 'var'),
    lr = 0.01; end
if ~exist('lr_decay', 'var'),
    lr_decay = 0.01; end
if ~exist('l2reg', 'var'),
    l2reg = 1e-5; end
if ~exist('dropout', 'var'),
    dropout = 0; end
if ~exist('maxiter', 'var'),
    maxiter = 100; end
if ~exist('saveiter', 'var'),
    saveiter = 10; end
if ~exist('batchsize', 'var'),
    batchsize = 32; end
if ~exist('dosample', 'var'),
    dosample = 0; end
if ~exist('std_init', 'var'),
    std_init = 0.01; end
l2reg = l2reg*batchsize;

recmodel = 'qzxy';
genmodel = 'pzx';
predmodel = 'pyxz';
modeltype = [recmodel '_' genmodel '_' predmodel];


rectype = 'relu';
pltype = 'sparse';
convtype = 'same';
nscale = size(numft_cls, 1);

savedir = [savedir '/' dataset '/'];
if ~exist(savedir, 'dir'),
    mkdir(savedir);
end
if ~exist('nettype', 'var'),
    nettype = 't1_cvae';
end
if ~isempty(strfind(nettype, 'gsnn')) || ~isempty(strfind(nettype, 'cnn')),
    alpha = 0;
end

% retrieve network parameters
[numhid, numft, numpool, stride, stotype, convtype, ...
    numhid_c, numft_c, numpool_c, stride_c, stotype_c, convtype_c] = retrieve_netparams(nettype);


% load data
[xtr, ytr, xval, yval, xts, yts, numlab] = data_loader(dataset, split, imsize, 0);

typein = 'real';
typeout = 'multinomial';
numvx = size(xtr, 3);
numvy = numlab;


% generate parameters
[params_xy2y, params_x2z, params_x2y] = mk_params(dataset, optgpu, savedir, recmodel, genmodel, ...
    do_recurrent, typein, typeout, numvx, numvy, l2reg, dropout, nsample, std_init, alpha, ...
    numhid, numft, numpool, stride, stotype, convtype, pltype, rectype, ...
    numhid_c, numft_c, numpool_c, stride_c, stotype_c, convtype_c, pltype, rectype);
params_x2y.numft_cls = numft_cls;

% multi-scale setting
net_pool.layers = {};
for j = 1:nscale-1,
    net_pool.layers{end+1} = struct('type', 'pool', 'method', 'max', ...
        'pool', abs([numpool_c(end-j+1), numpool_c(end-j+1)]), 'stride', abs(numpool_c(end-j+1)), 'pad', 0);
end


% generate network
[net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls] = mk_net(params_xy2y, params_x2z, params_x2y, nscale);


% -----------------------------------------------------------------------
%                                                      train network
% -----------------------------------------------------------------------

% define SGD parameters
params = struct(...
    'dataset', dataset, ...
    'optgpu', optgpu, ...
    'typein', typein, ...
    'typeout', typeout, ...
    'lr', lr, ...
    'lr_decay', lr_decay, ...
    'l2reg', l2reg, ...
    'savedir', savedir, ...
    'maxiter', maxiter, ...
    'saveiter', saveiter, ...
    'batchsize', batchsize, ...
    'nsample', nsample, ...
    'dosample', dosample, ...
    'droprate', droprate, ...
    'do_recurrent', do_recurrent, ...
    'alpha', alpha, ...
    'split', split, ...
    'verbose', 1, ...
    'modeltype', modeltype, ...
    'recmodel', recmodel, ...
    'genmodel', genmodel, ...
    'predmodel', predmodel, ...
    'nettype', nettype, ...
    'nscale', nscale);
params.xy2y = params_xy2y;
params.x2z = params_x2z;
params.x2y = params_x2y;
params.net_pool = net_pool;
if params.do_recurrent,
    params.fname = sprintf('%s_%s_rec_sc_%d_dr_%g_lr_%g_%g_l2r_%g_bs_%d_al_%g_nspl_%d', ...
        params.nettype, params.modeltype, params.nscale, params.droprate, ...
        params.lr, params.lr_decay, params.l2reg, params.batchsize, params.alpha, params.nsample);
else
    params.fname = sprintf('%s_%s_flat_sc_%d_dr_%g_lr_%g_%g_l2r_%g_bs_%d_al_%g_nspl_%d', ...
        params.nettype, params.modeltype, params.nscale, params.droprate, ...
        params.lr, params.lr_decay, params.l2reg, params.batchsize, params.alpha, params.nsample);
end
params.fname = [params.fname '_split_' num2str(params.split)];


% train network
fname = [params.fname '_iter_' num2str(params.maxiter)];
if exist([savedir '/' fname '.mat'], 'file'),
    fprintf('load pretrained weights\n');
    load([savedir '/' fname '.mat'], 'net_xy2z', 'net_z2y', 'net_x2z', 'net_x2y', 'net_cnn', 'net_cls', 'params', 't');
else
    [net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params, history, t] = ...
        dcgm_train(xtr, ytr, xval, yval, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params);
end


% -----------------------------------------------------------------------
%                                                       test network
% -----------------------------------------------------------------------

if strcmp(dotest, 'acc'),
    logdir = [logdir '/' dataset '/'];
    if ~exist(logdir, 'dir'),
        mkdir(logdir);
    end
    
    eval_iou = 0;
    if numlab == 2,
        eval_iou = 1;
    end
    
    batchsize = 64;
    res_z2y = [];
    res_x2z = [];
    res_x2y = cell(nscale, 1);
    res_cnn = cell(nscale, 1);
    res_cls = cell(nscale, 1);
    
    if optgpu,
        net_x2z = mcn_cpu2gpu(net_x2z);
        net_z2y = mcn_cpu2gpu(net_z2y);
        for j = 1:nscale,
            net_x2y{j} = mcn_cpu2gpu(net_x2y{j});
            net_cnn{j} = mcn_cpu2gpu(net_cnn{j});
            net_cls{j} = mcn_cpu2gpu(net_cls{j});
        end
    end
    
    % validation set
    nval = size(xval, 4);
    nbatch_val = ceil(nval/batchsize);
    
    err = 0;
    npixel = 0;
    err_cls = zeros(numlab, 1);
    npixel_cls = zeros(numlab, 1);
    if eval_iou,
        k = 0;
        iou_score = zeros(nval, 1);
    end
    
    for b = 1:nbatch_val,
        batchidx = (b-1)*batchsize+1:min(b*batchsize, nval);
        data = xval(:, :, :, batchidx);
        label = multi_output(yval(:, :, :, batchidx), numlab, 3, 'single');
        valid_idx = find(sum(label, 3) == 1);
        
        if optgpu,
            data = gpuArray(data);
            label = gpuArray(label);
        end
        
        net_cls{nscale}.layers{end}.data = label;
        
        [res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
            dcgm_inference(data, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
            params.do_recurrent, params.genmodel, params.predmodel);
        
        [~, pred] = max(gather(res_cls{nscale}(end-1).x), [], 3);
        [~, gt] = max(gather(label), [], 3);
        
        % pixel-wise
        err = err + sum(pred(valid_idx) ~= gt(valid_idx));
        npixel = npixel + length(valid_idx);
        
        % class-wise
        for j = 1:numlab,
            valid_idx_cls = intersect(find(gt == j), valid_idx);
            err_cls(j) = err_cls(j) + sum(pred(valid_idx_cls) ~= gt(valid_idx_cls));
            npixel_cls(j) = npixel_cls(j) + length(valid_idx_cls);
        end
        
        if eval_iou,
            for j = 1:length(batchidx),
                k = k + 1;
                cpred = gather(res_cls{nscale}(end-1).x(:, :, 2, j));
                cgt = gather(label(:, :, 2, j));
                iou_score(k) = compute_iou(cpred(:) > 0.5, cgt(:));
            end
        end
    end
    acc_val = 100*(1-err/npixel);
    acc_val_cls = 100*(1-mean(err_cls./npixel_cls));
    if eval_iou,
        iou_val = 100*mean(iou_score);
    end
    
    
    % test set
    nts = size(xts, 4);
    nbatch_ts = ceil(nts/batchsize);
    
    err = 0;
    npixel = 0;
    err_cls = zeros(numlab, 1);
    npixel_cls = zeros(numlab, 1);
    if eval_iou,
        k = 0;
        iou_score = zeros(nts, 1);
    end
    
    for b = 1:nbatch_ts,
        batchidx = (b-1)*batchsize+1:min(b*batchsize, nts);
        data = xts(:, :, :, batchidx);
        label = multi_output(yts(:, :, :, batchidx), numlab, 3, 'single');
        valid_idx = find(sum(label, 3) == 1);
        
        if optgpu,
            data = gpuArray(data);
            label = gpuArray(label);
        end
        
        net_cls{nscale}.layers{end}.data = label;
        
        [res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
            dcgm_inference(data, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
            params.do_recurrent, params.genmodel, params.predmodel);
        
        [~, pred] = max(gather(res_cls{nscale}(end-1).x), [], 3);
        [~, gt] = max(gather(label), [], 3);
        
        % pixel-wise
        err = err + sum(pred(valid_idx) ~= gt(valid_idx));
        npixel = npixel + length(valid_idx);
        
        % class-wise
        for j = 1:numlab,
            valid_idx_cls = intersect(find(gt == j), valid_idx);
            err_cls(j) = err_cls(j) + sum(pred(valid_idx_cls) ~= gt(valid_idx_cls));
            npixel_cls(j) = npixel_cls(j) + length(valid_idx_cls);
        end
        
        if eval_iou,
            for j = 1:length(batchidx),
                k = k + 1;
                cpred = gather(res_cls{nscale}(end-1).x(:, :, 2, j));
                cgt = gather(label(:, :, 2, j));
                iou_score(k) = compute_iou(cpred(:) > 0.5, cgt(:));
            end
        end
    end
    acc_ts = 100*(1-err/npixel);
    acc_ts_cls = 100*(1-mean(err_cls./npixel_cls));
    if eval_iou,
        iou_ts = 100*mean(iou_score);
    end
    
    if eval_iou,
        fprintf('[%d/%d] val: %g (iou: %g), ts: %g (iou: %g) (%s)\n', t(1), maxiter, acc_val, iou_val, acc_ts, iou_ts, fname);
        fid = fopen([logdir modeltype '.txt'], 'a+');
        fprintf(fid, '[%d/%d] val: %g (iou: %g), ts: %g (iou: %g) (%s)\n', t(1), maxiter, acc_val, iou_val, acc_ts, iou_ts, fname);
        fclose(fid);
    else
        fprintf('[%d/%d] val: %g (cls: %g), ts: %g (cls: %g) (%s)\n', t(1), maxiter, acc_val, acc_val_cls, acc_ts, acc_ts_cls, fname);
        fid = fopen([logdir modeltype '.txt'], 'a+');
        fprintf(fid, '[%d/%d] val: %g (cls: %g), ts: %g (cls: %g) (%s)\n', t(1), maxiter, acc_val, acc_val_cls, acc_ts, acc_ts_cls, fname);
        fclose(fid);
    end
    
elseif strcmp(dotest, 'll'),
    logdir = [logdir '/' dataset '/'];
    if ~exist(logdir, 'dir'),
        mkdir(logdir);
    end
    
    if ~isempty(strfind(nettype, 'cnn')),
        nsample = 1;
    else
        nsample = 50;
    end
    
    batchsize = 64;
    res_z2y = [];
    res_x2z = [];
    res_xy2z = [];
    res_x2y = cell(nscale, 1);
    res_cnn = cell(nscale, 1);
    res_cls = cell(nscale, 1);
    
    if optgpu,
        net_x2z = mcn_cpu2gpu(net_x2z);
        net_z2y = mcn_cpu2gpu(net_z2y);
        net_xy2z = mcn_cpu2gpu(net_xy2z);
        for j = 1:nscale,
            net_x2y{j} = mcn_cpu2gpu(net_x2y{j});
            net_cnn{j} = mcn_cpu2gpu(net_cnn{j});
            net_cls{j} = mcn_cpu2gpu(net_cls{j});
        end
    end
    
    % validation set
    nval = size(xval, 4);
    nbatch_val = ceil(nval/batchsize);
    logpy = zeros(nval, 1);
    
    for b = 1:nbatch_val,
        batchidx = (b-1)*batchsize+1:min(b*batchsize, nval);
        data = xval(:, :, :, batchidx);
        label = multi_output(yval(:, :, :, batchidx), numlab, 3, 'single');
        
        if optgpu,
            data = gpuArray(data);
            label = gpuArray(label);
        end
        
        net_cls{nscale}.layers{end}.data = label;
        
        [logpy(batchidx), res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
            dcgm_ll(data, label, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
            params.do_recurrent, params.alpha, params.recmodel, params.genmodel, params.predmodel, nsample);
    end
    logpy_val = mean(logpy);
    logpy_val_std = std(logpy);
    
    % test set
    nts = size(xts, 4);
    nbatch_ts = ceil(nts/batchsize);
    logpy = zeros(nts, 1);
    
    for b = 1:nbatch_ts,
        batchidx = (b-1)*batchsize+1:min(b*batchsize, nts);
        data = xts(:, :, :, batchidx);
        label = multi_output(yts(:, :, :, batchidx), numlab, 3, 'single');
        
        if optgpu,
            data = gpuArray(data);
            label = gpuArray(label);
        end
        
        net_cls{nscale}.layers{end}.data = label;
        
        [logpy(batchidx), res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
            dcgm_ll(data, label, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
            params.do_recurrent, params.alpha, params.recmodel, params.genmodel, params.predmodel, nsample);
    end
    logpy_ts = mean(logpy);
    logpy_ts_std = std(logpy);
    
    fprintf('[%d/%d] val: %g (std: %g), ts: %g (std: %g) (%s)\n', t(1), maxiter, logpy_val, logpy_val_std, logpy_ts, logpy_ts_std, fname);
    fid = fopen([logdir modeltype '_ll.txt'], 'a+');
    fprintf(fid, '[%d/%d] val: %g (std: %g), ts: %g (std: %g) (%s)\n', t(1), maxiter, logpy_val, logpy_val_std, logpy_ts, logpy_ts_std, fname);
    fclose(fid);
end

return;
