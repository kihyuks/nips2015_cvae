% baseline (single-scale, multi-scale)
demo_dcgm('lfw',1,'acc',split,128,1,0,'t1_cnn',[9 9],1,0,3e-4,1e-3,1e-7,0,500,500,32,0,0.01);
demo_dcgm('lfw',1,'acc',split,128,1,0,'t1_cnn',[3 3; 5 5; 9 9],1,0,3e-4,1e-3,1e-7,0,500,500,32,0,0.01);

% multi-scale
demo_dcgm('lfw',1,'acc',split,128,1,0,'t1_cnn',[3 3; 5 5; 9 9],1,0,3e-4,1e-3,1e-7,0,500,500,32,0,0.01);
demo_dcgm('lfw',1,'acc',split,128,1,0,'t1_gsnn',[3 3; 5 5; 9 9],1,0,3e-4,1e-3,1e-7,0,500,500,32,0,0.01);
demo_dcgm('lfw',1,'acc',split,128,1,0,'t1_cvae',[3 3; 5 5; 9 9],1,1,3e-4,1e-3,1e-7,0,500,500,32,0,0.01);
demo_dcgm('lfw',1,'acc',split,128,1,0,'t1_cvae',[3 3; 5 5; 9 9],1,0.5,3e-4,1e-3,1e-7,0,500,500,32,0,0.01);

% multi-scale, noise-injection
demo_dcgm('lfw',1,'acc',split,128,1,0.4,'t1_cnn',[3 3; 5 5; 9 9],1,0,3e-4,1e-3,1e-7,0,500,500,32,0,0.01);
demo_dcgm('lfw',1,'acc',split,128,1,0.4,'t1_gsnn',[3 3; 5 5; 9 9],1,0,3e-4,1e-3,1e-7,0,500,500,32,0,0.01);
demo_dcgm('lfw',1,'acc',split,128,1,0.4,'t1_cvae',[3 3; 5 5; 9 9],1,1,3e-4,1e-3,1e-7,0,500,500,32,0,0.01);
demo_dcgm('lfw',1,'acc',split,128,1,0.4,'t1_cvae',[3 3; 5 5; 9 9],1,0.5,3e-4,1e-3,1e-7,0,500,500,32,0,0.01);

