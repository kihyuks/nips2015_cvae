% log-likelihood of gaussian latent variables

function logpz = compute_softmax_loss(data, x, s, nsample)

nsample = numel(x)/numel(data);

% input already should have passed through sigmoid..
% loss for binary input
x = reshape(x, [size(data), nsample]);
x = min(max(x, 1e-7), 1-1e-7); % for numerical stability
logpz = -sum(sum(sum(bsxfun(@times, data, log(x)), 3), 1), 2);
logpz = -squeeze(logpz); % 1 x nsample

return;

