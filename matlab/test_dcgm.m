function test_dcgm(dataset, optgpu, imsize, do_recurrent, droprate, nettype, ...
    numft_cls, nsample, alpha, lr, lr_decay, l2reg, dropout, ...
    maxiter, saveiter, batchsize, dosample, std_init)

startup;
startup_matconvnet;

switch dataset,
    case 'cub',
        nsplit = 10;
    case 'lfw_goodparts',
        nsplit = 5;
end


% load trained models
nets = cell(1, nsplit);
for split = 1:nsplit,
    [net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params, t, fname] = ...
        dcgm(dataset, optgpu, 0, split, imsize, do_recurrent, ...
        droprate, nettype, numft_cls, nsample, alpha, lr, lr_decay, l2reg, ...
        dropout, maxiter, saveiter, batchsize, dosample, std_init);
    
    nets{split}.net_xy2z = net_xy2z;
    nets{split}.net_z2y = net_z2y;
    nets{split}.net_x2z = net_x2z;
    nets{split}.net_x2y = net_x2y;
    nets{split}.net_cnn = net_cnn;
    nets{split}.net_cls = net_cls;
    nets{split}.params = params;
end


% -----------------------------------------------------------------------
%                                                       test network
% -----------------------------------------------------------------------

% load data
[~, ~, ~, ~, xts, yts, numlab] = data_loader(dataset, split, imsize, 0);

fname = params.fname(1:strfind(params.fname, '_split')-1);
nscale = size(numft_cls, 1);

logdir = [logdir '/' dataset '/'];
if ~exist(logdir, 'dir'),
    mkdir(logdir);
end

batchsize = 64;
res_z2y = [];
res_x2z = [];
res_x2y = cell(nscale, 1);
res_cnn = cell(nscale, 1);
res_cls = cell(nscale, 1);


% test set
nts = size(xts, 4);
nbatch_ts = ceil(nts/batchsize);
pred_all = zeros(imsize, imsize, numlab, nts);

for split = 1:nsplit,
    net_x2z = nets{split}.net_x2z;
    net_z2y = nets{split}.net_z2y;
    net_x2y = nets{split}.net_x2y;
    net_cnn = nets{split}.net_cnn;
    net_cls = nets{split}.net_cls;
    
    if optgpu,
        net_x2z = mcn_cpu2gpu(net_x2z);
        net_z2y = mcn_cpu2gpu(net_z2y);
        for j = 1:nscale,
            net_x2y{j} = mcn_cpu2gpu(net_x2y{j});
            net_cnn{j} = mcn_cpu2gpu(net_cnn{j});
            net_cls{j} = mcn_cpu2gpu(net_cls{j});
        end
    end
    
    for b = 1:nbatch_ts,
        batchidx = (b-1)*batchsize+1:min(b*batchsize, nts);
        data = xts(:, :, :, batchidx);
        label = multi_output(yts(:, :, :, batchidx), numlab, 3, 'single');
        
        if optgpu,
            data = gpuArray(data);
            label = gpuArray(label);
        end
        
        net_cls{nscale}.layers{end}.data = label;
        
        [res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
            dcgm_inference(data, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_z2y, res_x2z, res_x2y, res_cnn, res_cls, ...
            params.do_recurrent, params.genmodel, params.predmodel);
        
        pred_all(:, :, :, batchidx) = pred_all(:, :, :, batchidx) + 1/nsplit*gather(res_cls{nscale}(end-1).x);
    end
end

eval_iou = 0;
if numlab == 2,
    eval_iou = 1;
end

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
    [~, pred] = max(pred_all(:, :, :, batchidx), [], 3);
    gt = yts(:, :, :, batchidx);
    label = multi_output(gt, numlab, 3, 'single');
    [~, gt] = max(label, [], 3);
    valid_idx = find(sum(label, 3) == 1);
    
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
            cpred = pred_all(:, :, 2, k);
            cgt = label(:, :, 2, j);
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
    fprintf('ts: %g (iou: %g) (%s)\n', acc_ts, iou_ts, fname);
    fid = fopen([logdir 'test.txt'], 'a+');
    fprintf(fid, 'ts: %g (iou: %g) (%s)\n', acc_ts, iou_ts, fname);
    fclose(fid);
else
    fprintf('ts: %g (cls: %g) (%s)\n', acc_ts, acc_ts_cls, fname);
    fid = fopen([logdir 'test.txt'], 'a+');
    fprintf(fid, 'ts: %g (cls: %g) (%s)\n', acc_ts, acc_ts_cls, fname);
    fclose(fid);
end

return;

