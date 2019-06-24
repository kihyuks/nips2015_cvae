function res = mcn_add_struct(res, res2)

assert(length(res) == length(res2));

for i = 1:length(res),
    if ~isempty(res2(i).filters),
        res(i).filters = res(i).filters + res2(i).filters;
        res(i).biases = res(i).biases + res2(i).biases;
    end
    if ~isempty(res2(i).filters_std),
        res(i).filters_std = res(i).filters_std + res2(i).filters_std;
        res(i).biases_std = res(i).biases_std + res2(i).biases_std;
    end
end

return