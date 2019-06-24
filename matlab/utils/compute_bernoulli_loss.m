% log-likelihood of bernoulli variables

function logpz = compute_bernoulli_loss(data, x, s, nsample)

nsample = numel(x)/numel(data);

% input already should have passed through sigmoid..
% loss for binary input
x = reshape(x, [size(data), nsample]);
logpz = -sum(sum(sum(bsxfun(@times, data, log(max(x, 1e-13))) + bsxfun(@times, 1-data, log(max(1-x, 1e-13))), 1), 2), 3);
logpz = -squeeze(logpz); % 1 x nsample

return;

