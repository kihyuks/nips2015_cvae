% VL_NNUPSAMPLE  CNN upsamplinng
%    Y = VL_NNUPSAMPLE(X) applies the unpooling operator to all
%    channels of the data X. X is a
%    SINGLE array of dimension H x W x D x N where (H,W) are the
%    height and width of the map stack, D is the image depth (number
%    of feature channels) and N the number of of images in the stack.
%
%    DZDX = VL_NNUPSAMPLE(X, DZDY) computes the derivatives of
%    the nework output Z w.r.t. the data X given the derivative DZDY
%    w.r.t the unpooling output Y.
%
%    VL_NNUPSAMPLE(..., 'option', value, ...) takes the following options:
%
%    Stride:: [1]
%      The output stride (unsampling factor). It can be either a
%      scalar for isotropic unsampling or a vector [STRIDEY
%      STRIDEX].
%
%    Method:: ['sparse']
%       Specify method of upsampling. It can be either 'sparse' (replace each 
%    entry of a feature map by an STRIDEY x STRIDEX block with the entry value
%    in the top left corner and zeros elsewhere) or 'dense' (replace each
%    entry of a feature map by an STRIDEY x STRIDEX block with the same entry
%    value in the top left corner).
%
%
%    The output a is a SINGLE array of dimension YH x YW x K x N of N
%    images with K challens and size:
%
%      YH = H * STRIDEY,
%      YW = W * STRIDEX.
%
%    The derivative DZDY has the same dimension of the output Y and
%    the derivative DZDX has the same dimension as the input X.

