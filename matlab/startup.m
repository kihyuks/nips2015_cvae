% startup
addpath utils/;
addpath mcn/;
addpath data/;
addpath dcgm/;
addpath scripts/;

ccd = pwd;
logdir = [ccd '/log/'];
if ~exist(logdir, 'dir'),
    mkdir(logdir);
end

savedir = [ccd '/results/'];
if ~exist(savedir, 'dir'),
    mkdir(savedir);
end

datadir = [ccd '/data/'];
