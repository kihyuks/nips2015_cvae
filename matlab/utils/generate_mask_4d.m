function mask = generate_mask_4d(dim, dimmask, batchsize)

mask = zeros([dim, 1, batchsize]);

if dim(1) == dimmask(1) || dim(2) == dimmask(2),
    mask = 1-mask;
else
    rowidx = randi(dim(1)-dimmask(1), batchsize);
    colidx = randi(dim(2)-dimmask(2), batchsize);
    
    for i = 1:batchsize,
        mask(rowidx(i):rowidx(i)+dimmask(1)-1, colidx(i):colidx(i)+dimmask(2)-1, :, i) = 1;
    end
end

return