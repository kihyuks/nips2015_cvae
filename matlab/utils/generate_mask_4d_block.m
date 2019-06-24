function mask = generate_mask_4d_block(dim, droprate, blocksize, batchsize, blocktype)

if ~exist('droprate', 'var'),
    droprate = 0.5;
end
if ~exist('blocksize', 'var'),
    blocksize = 1;
end
if ~exist('blocktype', 'var'),
    blocktype = 'multiple';
end

switch blocktype,
    case 'multiple',
        if blocksize == 1,
            mask = zeros([dim, 1, batchsize]);
            mask = rand(size(mask)) < droprate;
        else
            % no overlap
            mask_rs = zeros([dim/blocksize, 1, batchsize]);
            mask_rs = rand(size(mask_rs)) < droprate;
            
            mask = zeros([dim, 1, batchsize]);
            mask(1:blocksize:end, 1:blocksize:end, :, :) = mask_rs;
            mask = convn(double(mask), ones(blocksize), 'same');
            mask = mask > 0;
        end
        
    case 'single',
        % no need for droprate, but blocksize matters
        mask = zeros([dim, 1, batchsize]);
        for i = 1:batchsize,
            r = randsample(1:dim(1)-blocksize+1, 1);
            c = randsample(1:dim(1)-blocksize+1, 1);
            mask(r:r+blocksize-1,c:c+blocksize-1,:,i) = 1;
        end
end

return