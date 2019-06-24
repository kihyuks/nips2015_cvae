function [numhid, numft, numpool, stride, stotype, convtype, ...
    numhid_c, numft_c, numpool_c, stride_c, stotype_c, convtype_c] = retrieve_netparams(nettype)

switch nettype,
    case {'t1_cnn', 't1_gsnn', 't1_cvae'},
        numhid = [64; 96; 128; 32; 96; 64; 48; 48];
        numft = [9 9; 5 5; 3 3; 3 3; 3 3; 5 5; 5 5; 5 5; 5 5];
        numpool = [2; 2; 2; 1; -2; -2; -2; -2];
        stride = [2; 1; 1; 1; 1; 1; 1; 1; 1];
        convtype = 'same';
        numhid_c = [64; 96; 128; 128; 96; 64; 48];
        numft_c = [9 9; 5 5; 3 3; 3 3; 3 3; 5 5; 5 5];
        numpool_c = [2; 2; 1; 1; -2; -2; -2];
        stride_c = [2; 1; 1; 1; 1; 1; 1];
        convtype_c = 'same';
        if ~isempty(strfind(nettype, 'cnn')),
            stotype = [0; 0; 0; 1; 0; 0; 0; 0];
            stotype_c = [0; 0; 0; 0; 0; 0; 0];
        elseif ~isempty(strfind(nettype, 'gsnn')) || ~isempty(strfind(nettype, 'cvae')),
            stotype = [0; 0; 0; 3; 0; 0; 0; 0];
            stotype_c = [0; 0; 0; 2; 0; 0; 0];
        end
        
    case {'t1ft_cnn', 't1ft_gsnn', 't1ft_cvae'},
        numhid = [64; 96; 128; 32; 96; 64; 48; 48];
        numft = [9 9; 5 5; 3 3; 3 3; 3 3; 5 5; 5 5; 5 5; 5 5];
        numpool = [2; 2; 2; 1; -2; -2; -2; -2];
        stride = [2; 1; 1; 1; 1; 1; 1; 1; 1];
        convtype = 'same';
        numhid_c = [64; 96; 128; 128; 96; 64; 48];
        numft_c = [9 9; 5 5; 3 3; 3 3; 3 3; 5 5; 5 5];
        numpool_c = [2; 2; 1; 1; -2; -2; -2];
        stride_c = [2; 1; 1; 1; 1; 1; 1];
        convtype_c = 'same';
        if ~isempty(strfind(nettype, 'cnn')),
            stotype = [0; 0; 0; 1; 0; 0; 0; 0];
            stotype_c = [0; 0; 0; 0; 0; 0; 0];
        elseif ~isempty(strfind(nettype, 'gsnn')) || ~isempty(strfind(nettype, 'cvae')),
            stotype = [0; 0; 0; 3; 0; 0; 0; 0];
            stotype_c = [0; 0; 0; 2; 0; 0; 0];
        end
        
    case {'gc_cnn', 'gc_gsnn', 'gc_cvae'},
        % gradient check
        numhid = [1; 1; 1];
        numft = [3 3; 3 3; 1 1; 3 3];
        numpool = [1; -1; -1];
        stride = [1; 1; 1; 1];
        convtype = 'same';
        numhid_c = [1; 1; 1];
        numft_c = [3 3; 3 3; 1 1; 3 3];
        numpool_c = [1; -1; -1];
        stride_c = [1; 1; 1; 1];
        convtype_c = 'same';
        if ~isempty(strfind(nettype, 'cnn')),
            stotype = [1; 0; 0];
            stotype_c = [0; 0; 0];
        elseif ~isempty(strfind(nettype, 'gsnn')) || ~isempty(strfind(nettype, 'cvae')),
            stotype = [3; 0; 0];
            stotype_c = [2; 0; 0];
        end
        
    case {'t2_cnn', 't2_gsnn', 't2_cvae'},
        numhid = [48; 48; 64; 64; 256; 64; 64; 64; 48; 48];
        numft = [5 5; 5 5; 8 8; 1 1; 1 1; 1 1; 8 8; 5 5; 5 5; 5 5; 5 5];
        numpool = [2; 2; 1; 1; 1; 1; -2; -2; -2; -2];
        stride = [2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1];
        convtype = {'same';'same';'fconv_valid';'same';'same';'same';'fconv_full';'same';'same';'same';'same'};
        numhid_c = [64; 96; 128; 128; 96; 64; 48];
        numft_c = [9 9; 5 5; 3 3; 3 3; 3 3; 5 5; 5 5];
        numpool_c = [2; 2; 1; 1; -2; -2; -2];
        stride_c = [2; 1; 1; 1; 1; 1; 1];
        convtype_c = 'same';
        if ~isempty(strfind(nettype, 'cnn')),
            stotype = [0; 0; 0; 0; 1; 0; 0; 0; 0; 0];
            stotype_c = [0; 0; 0; 0; 0; 0; 0; 0; 0];
        elseif ~isempty(strfind(nettype, 'gsnn')) || ~isempty(strfind(nettype, 'cvae')),
            stotype = [0; 0; 0; 0; 3; 0; 0; 0; 0; 0];
            stotype_c = [0; 0; 0; 0; 2; 0; 0; 0; 0];
        end
        
    case {'t3_cnn', 't3_gsnn', 't3_cvae'},
        numhid = [64; 96; 128; 32; 96; 64; 48; 48; 48];
        numft = [9 9; 5 5; 3 3; 3 3; 3 3; 5 5; 5 5; 5 5; 5 5; 5 5];
        numpool = [2; 2; 2; 1; -2; -2; -2; -2; -2];
        stride = [2; 2; 1; 1; 1; 1; 1; 1; 1; 1];
        convtype = 'same';
        numhid_c = [64; 96; 128; 128; 96; 64; 48];
        numft_c = [9 9; 5 5; 3 3; 3 3; 3 3; 5 5; 5 5];
        numpool_c = [2; 2; 1; 1; -2; -2; -2];
        stride_c = [2; 1; 1; 1; 1; 1; 1];
        convtype_c = 'same';
        if ~isempty(strfind(nettype, 'cnn')),
            stotype = [0; 0; 0; 1; 0; 0; 0; 0; 0];
            stotype_c = [0; 0; 0; 0; 0; 0; 0];
        elseif ~isempty(strfind(nettype, 'gsnn')) || ~isempty(strfind(nettype, 'cvae')),
            stotype = [0; 0; 0; 3; 0; 0; 0; 0; 0];
            stotype_c = [0; 0; 0; 2; 0; 0; 0];
        end
    case {'t4_cnn', 't4_gsnn', 't4_cvae'},
        numhid = [64; 96; 128; 32; 96; 64; 48; 48; 48];
        numft = [9 9; 5 5; 3 3; 3 3; 3 3; 5 5; 5 5; 5 5; 5 5; 5 5];
        numpool = [2; 2; 2; 1; -2; -2; -2; -2; -2];
        stride = [2; 2; 1; 1; 1; 1; 1; 1; 1; 1];
        convtype = 'same';
        numhid_c = [64; 96; 128; 128; 96; 64; 48; 48; 48];
        numft_c = [9 9; 5 5; 3 3; 3 3; 3 3; 5 5; 5 5; 5 5; 5 5; 5 5];
        numpool_c = [2; 2; 2; 1; -2; -2; -2; -2; -2];
        stride_c = [2; 2; 1; 1; 1; 1; 1; 1; 1; 1];
        convtype_c = 'same';
        if ~isempty(strfind(nettype, 'cnn')),
            stotype = [0; 0; 0; 1; 0; 0; 0; 0; 0];
            stotype_c = [0; 0; 0; 0; 0; 0; 0; 0; 0];
        elseif ~isempty(strfind(nettype, 'gsnn')) || ~isempty(strfind(nettype, 'cvae')),
            stotype = [0; 0; 0; 3; 0; 0; 0; 0; 0];
            stotype_c = [0; 0; 0; 2; 0; 0; 0; 0; 0];
        end
end

return;