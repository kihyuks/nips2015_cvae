function [net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params, history, t] = ...
    dcgm_train(xtr, ytr, xval, yval, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params)


% filename to save
params.fname_save = sprintf('%s_iter_%d', params.fname, params.maxiter);

nscale = length(net_cls);

% make sure all data are in single-precision
xtr = single(xtr);
xval = single(xval);
ytr = double(ytr);      % label vector, 1 to L
yval = double(yval);    % label vector, 1 to L


% train network
[net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params, history, t] = ...
    dcgm_train_adam(xtr, ytr, xval, yval, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params);


% gpu -> cpu
net_xy2z = mcn_gpu2cpu(net_xy2z);
net_x2z = mcn_gpu2cpu(net_x2z);
net_z2y = mcn_gpu2cpu(net_z2y);
for j = 1:nscale,
    net_x2y{j} = mcn_gpu2cpu(net_x2y{j});
    net_cnn{j} = mcn_gpu2cpu(net_cnn{j});
    net_cls{j} = mcn_gpu2cpu(net_cls{j});
end

return;


% -----------------------------------------------------------------------
% 	trainer for conditional VAE with AdaM optimization
% -----------------------------------------------------------------------

function [net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params, history, t] = dcgm_train_adam...
    (xtr, ytr, xval, yval, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params)

rng('default');

nscale = length(net_cls);

net_xy2z_best = net_xy2z;
net_z2y_best = net_z2y;
net_x2z_best = net_x2z;
net_x2y_best = net_x2y;
net_cnn_best = net_cnn;
net_cls_best = net_cls;
tstar = 0;

% convert to gpu variables
if params.optgpu,
    net_xy2z = mcn_cpu2gpu(net_xy2z);
    net_x2z = mcn_cpu2gpu(net_x2z);
    net_z2y = mcn_cpu2gpu(net_z2y);
    for j = 1:nscale,
        net_x2y{j} = mcn_cpu2gpu(net_x2y{j});
        net_cnn{j} = mcn_cpu2gpu(net_cnn{j});
        net_cls{j} = mcn_cpu2gpu(net_cls{j});
    end
end

% filename to save
fname_mat = sprintf('%s/%s.mat', params.savedir, params.fname_save);
disp(params);


% -----------------------------------------------------------------------
%                                                      train networks
% -----------------------------------------------------------------------

% start sgd training
maxiter = params.maxiter;
batchsize = params.batchsize;
ntr = size(xtr, 4);
nbatch = min(floor(min(ntr, 100000)/batchsize), 20); % 100000 samples per iteration
droprate = params.droprate;

if exist('xval', 'var') && ~isempty(xval),
    doval = 1;
    nval = size(xval, 4);
    nbatch_val = ceil(nval/batchsize);
    val_err = 100;
else
    doval = 0;
end


% monitoring variables
history = struct;
history.lb = zeros(maxiter, 1);
history.ll = zeros(maxiter, 1);
history.KL = zeros(maxiter, 1);
if doval,
    history.val.err_cls = zeros(maxiter, 1);
    history.val.err = zeros(maxiter, 1);
    history.val.lb = zeros(maxiter, 1);
    history.val.ll = zeros(maxiter, 1);
    history.val.KL = zeros(maxiter, 1);
end

b1 = 0.1;
b2 = 0.001;
ii = 1;

nround = ceil(10000/nbatch);

res_xy2z = [];
res_x2z = [];
res_z2y = [];
res_x2z2y = [];
res_x2y = cell(nscale, 1);
res_cnn = cell(nscale, 1);
res_cls = cell(nscale, 1);
res_x2cls = cell(nscale, 1);

imsize = [size(xtr, 1), size(xtr, 2)];
numlab = length(unique(ytr(:)));

for t = 1:maxiter,
    % at every training epoch, we use different random seed
    % (this is for retraining purpose)
    rng(t);
    
    KL_epoch = zeros(nbatch, 1);
    ll_epoch = zeros(nbatch, 1);
    lb_epoch = zeros(nbatch, 1);
    err_epoch = zeros(nbatch, 1);
    
    randidx = randperm(ntr);
    
    % decay as if it is momentum training
    if t > nround,
        lr = params.lr/(1+params.lr_decay*(t-nround));
    else
        lr = params.lr;
    end
    
    tS = tic;
    
    label = cell(nscale, 1);
    for b = 1:nbatch,
        % construct minibatch
        batchidx = randidx((b-1)*batchsize+1:b*batchsize);
        data = xtr(:, :, :, batchidx);
        label_mult = multi_output(ytr(:, :, :, batchidx), numlab, 3, 'single');
        
        % multi-scale
        res_pool = mcn_ff(params.net_pool, label_mult, [], []);
        for j = 1:nscale,
            label{j} = bsxfun(@rdivide, res_pool(end-j+1).x, sum(res_pool(end-j+1).x, 3));
        end
        
        if params.dosample,
            if strcmp(params.typein, 'binary') || strcmp(params.typein, 'bernoulli'),
                data = single(rand(size(data)) < data);
            elseif strcmp(params.typein, 'multinomial') || strcmp(params.typein, 'softmax'),
                data = realize_mult(data, 3);
            end
        end
        
        % mask (generate mask with size < imsize*droprate)
        boxsize = round(imsize*rand(1)*droprate);
        mask = generate_mask_4d(imsize, boxsize, length(batchidx));
        data = bsxfun(@times, data, 1-mask);
        
        if params.optgpu,
            data = gpuArray(data);
            for j = 1:nscale,
                label{j} = gpuArray(label{j});
            end
        end
        
        % compute cost and gradient
        for j = 1:nscale,
            net_cls{j}.layers{end}.data = label{j};
        end
        
        [cost, res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, KL] = ...
            dcgm_cost(data, label{nscale}, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
            res_xy2z, res_z2y, res_x2z, res_x2y, res_cnn, res_cls, res_x2z2y, res_x2cls, ...
            params.do_recurrent, params.alpha, params.recmodel, params.genmodel, params.predmodel, 0);
        
        % costs
        ll_epoch(b) = -gather(res_cls{nscale}(end).x)/batchsize;
        KL_epoch(b) = -gather(KL)/batchsize;
        lb_epoch(b) = ll_epoch(b) + KL_epoch(b);
        
        if strcmp(params.typeout, 'multinomial') || strcmp(params.typeout, 'softmax'),
            valid_idx = find(sum(label{nscale}, 3) == 1);
            [~, pred] = max(gather(res_cls{nscale}(end-1).x), [], 3);
            [~, gt] = max(gather(label{nscale}), [], 3);
            err_epoch(b) = sum(pred(valid_idx) ~= gt(valid_idx))/numel(pred(valid_idx));
        end
        
        % update parameter
        if params.alpha > 0,
            net_xy2z = update_adam(net_xy2z, res_xy2z, lr, b1, b2, ii);
        end
        if params.do_recurrent,
            net_x2z = update_adam(net_x2z, res_x2z, lr, b1, b2, ii);
            net_z2y = update_adam(net_z2y, res_z2y, lr, b1, b2, ii);
        elseif any(strfind(params.nettype, 'cvae')),
            net_x2z = update_adam(net_x2z, res_x2z, lr, b1, b2, ii);
            net_z2y = update_adam(net_z2y, res_z2y, lr, b1, b2, ii);
        end
        for j = 1:nscale,
            net_x2y{j} = update_adam(net_x2y{j}, res_x2y{j}, lr, b1, b2, ii);
            net_cnn{j} = update_adam(net_cnn{j}, res_cnn{j}, lr, b1, b2, ii);
            net_cls{j} = update_adam(net_cls{j}, res_cls{j}, lr, b1, b2, ii);
        end
        ii = ii + 1;
    end
    
    history.ll(t) = double(sum(ll_epoch))/nbatch;
    history.KL(t) = double(sum(KL_epoch))/nbatch;
    history.lb(t) = double(sum(lb_epoch))/nbatch;
    history.err(t) = 100*double(sum(err_epoch))/nbatch;
    
    if doval,
        history.val.err(t) = 0;
        history.val.lb(t) = 0;
        err = 0;
        npixel = 0;
        err_cls = zeros(numlab, 1);
        npixel_cls = zeros(numlab, 1);
        logpy = zeros(nval, 1);
        
        % validation
        for b = 1:nbatch_val,
            batchidx = (b-1)*batchsize+1:min(b*batchsize, nval);
            data = xval(:, :, :, batchidx);
            label = multi_output(yval(:, :, :, batchidx), numlab, 3, 'single');
            valid_idx = find(sum(label, 3) == 1);
            
            if params.optgpu,
                data = gpuArray(data);
                label = gpuArray(label);
            end
            
            net_cls{nscale}.layers{end}.data = label;
            if strcmp(params.typeout, 'multinomial') || strcmp(params.typeout, 'softmax'),
                [res_z2y, res_x2z, res_x2y, res_cnn, res_cls] = ...
                    dcgm_inference(data, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, ...
                    res_z2y, res_x2z, res_x2y, res_cnn, res_cls, params.do_recurrent, ...
                    params.genmodel, params.predmodel);
                
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
            end
        end
        
        if strcmp(params.typeout, 'multinomial') || strcmp(params.typeout, 'softmax'),
            history.val.err(t) = 100*err/npixel;
            history.val.err_cls(t) = 100*(mean(err_cls./npixel_cls));
            
            if ~isfield(params, 'cv'),
                cur_err = history.val.err(t);
            else
                if strcmp(params.cv, 'pixel'),
                    cur_err = history.val.err(t);
                elseif strcmp(params.cv, 'class'),
                    cur_err = history.val.err_cls(t);
                end
            end
            
            if cur_err < val_err,
                tstar = t;
                val_err = cur_err;
                net_xy2z_best = net_xy2z;
                net_z2y_best = net_z2y;
                net_x2z_best = net_x2z;
                net_x2y_best = net_x2y;
                net_cnn_best = net_cnn;
                net_cls_best = net_cls;
            end
        end
    end
    
    tE = toc(tS);
    if params.verbose,
        if strcmp(params.typeout, 'multinomial') || strcmp(params.typeout, 'softmax'),
            fprintf('epoch %d (train):\t lb = %g, ll = %g, KL = %g, err = %g (time = %g)\n', ...
                t, history.lb(t), history.ll(t), history.KL(t), history.err(t), tE);
            if doval,
                fprintf('epoch %d (val):\t err = %g, err-cls = %g\n', t, history.val.err(t), history.val.err_cls(t));
            end
        elseif isfield(params, 'cv') && strcmp(params.cv, 'cll'),
            fprintf('epoch %d (train):\t lb = %g, ll = %g, KL = %g (time = %g)\n', ...
                t, history.lb(t), history.ll(t), history.KL(t), tE);
            if doval,
                fprintf('epoch %d (val):\t cll = %g (best = %g @ t = %d)\n', t, history.val.lb(t), val_cll, tstar);
            end
        else
            fprintf('epoch %d (train):\t lb = %g, ll = %g, KL = %g (time = %g)\n', ...
                t, history.lb(t), history.ll(t), history.KL(t), tE);
            if doval,
                fprintf('epoch %d (val):\t lb = %g\n', t, history.val.lb(t));
            end
        end
    end
    
    % save parameters every few epochs
    if mod(t, params.saveiter) == 0,
        if ~params.verbose,
            if strcmp(params.typeout, 'multinomial') || strcmp(params.typeout, 'softmax'),
                fprintf('epoch %d (train):\t lb = %g, ll = %g, KL = %g, err = %g (time = %g)\n', ...
                    t, history.lb(t), history.ll(t), history.KL(t), history.err(t), tE);
                if doval,
                    fprintf('epoch %d (val):\t err = %g, err-cls = %g\n', t, history.val.err(t), history.val.err_cls(t));
                end
            elseif isfield(params, 'cv') && strcmp(params.cv, 'cll'),
                fprintf('epoch %d (train):\t lb = %g, ll = %g, KL = %g (time = %g)\n', ...
                    t, history.lb(t), history.ll(t), history.KL(t), tE);
                if doval,
                    fprintf('epoch %d (val):\t cll = %g (best = %g @ t = %d)\n', t, history.val.lb(t), val_cll, tstar);
                end
            else
                fprintf('epoch %d (train):\t lb = %g, ll = %g, KL = %g (time = %g)\n', ...
                    t, history.lb(t), history.ll(t), history.KL(t), tE);
                if doval,
                    fprintf('epoch %d (val):\t lb = %g\n', t, history.val.lb(t));
                end
            end
        end
        fname_temp = sprintf('%s/%s_iter_%d.mat', params.savedir, params.fname, t);
        if exist('val_err', 'var') || exist('val_cll', 'var'),
            save_params(fname_temp, net_xy2z_best, net_z2y_best, net_x2z_best, net_x2y_best, net_cnn_best, net_cls_best, params, [tstar t], history);
        else
            save_params(fname_temp, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params, t, history);
        end
        
        fprintf('%s/%s_iter_%d.mat\n', params.savedir, params.fname, t);
    end
end

% save parameters and visualizations
if exist('val_err', 'var') || exist('val_cll', 'var'),
    [net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls] = ...
        save_params(fname_mat, net_xy2z_best, net_z2y_best, net_x2z_best, net_x2y_best, net_cnn_best, net_cls_best, params, [tstar t], history);
    t = [tstar t];
else
    [net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls] = ...
        save_params(fname_mat, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params, t, history);
end

return;


function [net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls] = ...
    save_params(fname, net_xy2z, net_z2y, net_x2z, net_x2y, net_cnn, net_cls, params, t, history)

nscale = size(net_cls);

net_xy2z = mcn_gpu2cpu(net_xy2z);
net_x2z = mcn_gpu2cpu(net_x2z);
net_z2y = mcn_gpu2cpu(net_z2y);
for j = 1:nscale,
    net_x2y{j} = mcn_gpu2cpu(net_x2y{j});
    net_cnn{j} = mcn_gpu2cpu(net_cnn{j});
    net_cls{j} = mcn_gpu2cpu(net_cls{j});
end

save(fname, 'net_xy2z', 'net_z2y', 'net_x2z', 'net_x2y', 'net_cnn', 'net_cls', 'params', 't', 'history');

return;
