% script generate_datasets_ncomms2020
% Generating datasets for NComms paper

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%     Generate convnet network metadata     %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate metadata for the different networks
for seed=0:4; generate_convnet_model_metadata(1, nan, seed); end % alexnet
epoch = 0; generate_convnet_model_metadata(1, epoch, 0);
generate_convnet_model_metadata(3);                              % resnet50
for seed=0:4; generate_convnet_model_metadata(5, nan, seed); end % vgg16
epoch = 0; generate_convnet_model_metadata(5, epoch);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%     1D manifolds data-set     %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RANGE_FACTORS = [0.125, 0.25, 0.5, 1, 2, 4, 8, 12, 16];
SAMPLES = [5, 9, 15, 27, 51, 101, 201, 201, 201];

% Create dataset for alexnet
for i=1:9
    for id=1:28; generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 1, id); end 
    generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 1, 1:28);                
end
i=9; 
% Generate control data with random projections
for id=1:28; generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 1, id, 4, true); end 
generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 1, 1:28, 4, true);

% Create results per epoch for alexnet
i=9; epoch = 0; 
for id=1:28; generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 1, id, 4, false, epoch); end
generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 1, 1:28, 4, false, epoch);

% Create dataset for resnet50
for i=1:6
    for id=1:28; generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 3, id); end 
    generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 3, 1:28);                
end
for i=7:9
    for id=1:7*128; generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 3, id, 128); end
    generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 3, 1:7*128, 128);
end

% Create dataset for vgg16
for i=1:9
    for id=1:28; generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 5, id); end
    generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 5, 1:28);
end
i=9; epoch=0;
for id=1:28; generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 5, id, 4, false, epoch); end 
generate_convnet_one_dimensional_change(RANGE_FACTORS(i), 128, SAMPLES(i), 4, 1:28, 4, false, epoch);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%     Randomly sampled manifolds data-set     %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RANGE_FACTORS = [0.125, 0.25, 0.5, 1, 2, 4, 8, 12, 16];
SAMPLES_2D = [51, 101, 201, 401, 801, 1601, 3201, 3201, 3201];

% Visualize data
show_imagenet_random_change(128, 8, 2, 0)
show_imagenet_random_change(128, 16, 2, 0)

% Generate 2D manifolds for alexnet with P=64
n_batches = [4, 4, 4, 4, 4, 4, 64, 64, 64];
for i=1:9
    for id=1:n_batches(i)*2; generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, n_batches(i), id); end
    generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, n_batches(i), 1:n_batches(i)*2);
end
% Generate 2D manifolds for alexnet with P=64 (control, different neurons)
i=9; n_batches=64;
for seed=1:4
    for id=1:n_batches*2; generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, n_batches, id, nan, seed); end
    generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, n_batches, 1:n_batches*2, nan, seed);
end
% Generate 2D manifolds for alexnet with P=64 (control, different objects)
i=9; n_batches=64;
for seed=1:4
    for id=1:n_batches*2; generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, n_batches, id, nan, 0, seed); end
    generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, n_batches, 1:n_batches*2, nan, 0, seed);
end
i=9; n_batches=64; epoch=0;
for seed=1:4
    for id=1:n_batches*2; generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, n_batches, id, epoch, 0, seed); end
    generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, n_batches, 1:n_batches*2, epoch, 0, seed);
end

% Generate 2D representation for vgg16 
n_batches = 4;
for i=1:3
    for id=1:2*n_batches; generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, n_batches, id); end   % @hesed
    generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, n_batches, 1:2*n_batches);                  % @keter
end
for i=4:9
    n_batches = 64; MAX_BATCH = 2*n_batches*ceil(SAMPLES_2D(i)/201);
    for id=1:MAX_BATCH; generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, n_batches, id, nan, 201); end
    generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, n_batches, 1:MAX_BATCH, nan, 201);
end
% Generate 2D representation for vgg16 (control, different neurons) 
i=9; n_batches = 64; MAX_BATCH = 2*n_batches*ceil(SAMPLES_2D(i)/201);
for seed=1:4
    for id=1:MAX_BATCH; generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, n_batches, id, nan, 201, seed); end
    generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, n_batches, 1:MAX_BATCH, nan, 201, seed);
end
% Generate 2D representation for vgg16 (control, different objects) 
i=9; n_batches = 64; MAX_BATCH = 2*n_batches*ceil(SAMPLES_2D(i)/201);
for seed=1:4
    for id=1:MAX_BATCH; generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, n_batches, id, nan, 201, 0, seed); end
    generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, n_batches, 1:MAX_BATCH, nan, 201, 0, seed);
end

% Generate 2D representation for googlenet 
i=8; n_batches = 64; MAX_BATCH = 2*n_batches*ceil(SAMPLES_2D(i)/801);
for id=1:MAX_BATCH; generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 2, 2, n_batches, id, nan, 801); end
generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 2, 2, n_batches, 1:MAX_BATCH, nan, 801);

% Generate 2D representation for resnet50
for i=1:9
    n_batches = 64; MAX_BATCH = 2*n_batches*ceil(SAMPLES_2D(i)/201);
    for id=1:MAX_BATCH; generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 3, 2, n_batches, id, nan, 201); end
    generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 3, 2, n_batches, 1:MAX_BATCH, nan, 201);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Randomly sampled manifolds data-set     
% Theoretical capacity using low-rank approximation of correlations matrix 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate 2D representation for alexnet at different training epochs
i=8; n_batches = 64; 
for epoch=[0, 1, 2, 3, 4, 5, 10, 50, 90, inf]
    for id=1:n_batches*2; generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, n_batches, id, epoch); end
    generate_convnet_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, n_batches, 1:n_batches*2, epoch);
end

% Generate 2D representation for vgg16 at different training epochs
i=8; n_batches = 64; MAX_BATCH = 2*n_batches*ceil(SAMPLES_2D(i)/201);
epoch=0;
for id=1:MAX_BATCH; generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, n_batches, id, epoch, 201); end
generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, n_batches, 1:MAX_BATCH, epoch, 201);

% Generate 2D representation for resnet18 at different training epochs 
i=8; n_batches = 64; MAX_BATCH = 2*n_batches*ceil(SAMPLES_2D(i)/801);
for epoch=[0, 1, 2, 3, 4, 5, 10, 50, 90, nan]
    for id=1:MAX_BATCH; generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 4, 2, n_batches, id, epoch, 801); end
    generate_convnet_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 4, 2, n_batches, 1:MAX_BATCH, epoch, 801);
end
