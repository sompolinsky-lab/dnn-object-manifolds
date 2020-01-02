% script generate_results_ncomms2020
% Generating results for NComms paper
return; % the code is provided mainly as reference and should not be run as-is

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%     1D manifolds data-set     %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RANGE_FACTORS = [0.125, 0.25, 0.5, 1, 2, 4, 8, 12, 16];
SAMPLES = [5, 9, 15, 27, 51, 101, 201, 201, 201];
init_imagenet;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate capacity for 1d data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(RANGE_FACTORS)
    % Numericallay calculate capacity for alexnet
    for run_id=1:7*22; check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, run_id); end
    check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 1:7*22);                          
    % Numericallay calculate capacity with randomized centers for alexnet
    for run_id=1:7*22; check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 2, '', run_id); end
    check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 2, '', 1:7*22);

    % Numericallay calculate capacity with randomized manifolds for alexnet
    for run_id=1:7*22; check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 3, '', run_id); end   
    check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 3, '', 1:7*22);                          
    % Numericallay calculate capacity with randomized axes for alexnet
    for run_id=1:7*22; check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 5, '', run_id); end
    check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 5, '', 1:7*22);

    % Numericallay calculate capacity with permuted axes for alexnet
    for run_id=1:7*22; check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 7, '', run_id); end
    check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 7, '', 1:7*22);

    % Numericallay calculate capacity for resnet
    for run_id=1:7*176; check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 3, 0, run_id); end
    check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 3, 0, 1:7*176);

    % Numericallay calculate capacity for vgg16
    for run_id=1:7*38; check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 5, 0, run_id); end
    check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 5, 0, 1:7*38);
end

% Numericallay calculate capacity for alexnet before training
i = 9; epoch=0;
for run_id=1:7*21; check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, run_id, epoch); end
check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 1:7*21, epoch);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data manipulation to verify the effect on R and D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i = 9;
for run_id=1:7*22; check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 5, run_id); end
check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 5, 1:7*21);

D = nan;
for center_norm_factor = [0.9, 0.95, 1., 1.05, 1.1]
for run_id=1:7*22; check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, run_id, nan, D, center_norm_factor); end
check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 1:7*21, nan, D, center_norm_factor);
end

center_norm_factor=1;
for D=[10, 20, 30, 40, 50]
for run_id=1:7*22; check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, run_id, nan, D, center_norm_factor); end
check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 1:7*21, nan, D, center_norm_factor);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Control experiments (aka "causality vs correlations" discussion)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Numericallay calculate capacity for spheres with the same geometry as 1d alexnet manifolds 
for i=1:9
    for id=1:154
        check_convnet_capacity_one_dimensional_change2_spheres(128, RANGE_FACTORS(i), SAMPLES(i), 1, false, id);
        check_convnet_capacity_one_dimensional_change2_spheres(128, RANGE_FACTORS(i), SAMPLES(i), 1, true, id);
    end
    check_convnet_capacity_one_dimensional_change2_spheres(128, RANGE_FACTORS(i), SAMPLES(i), 1, false, 1:154);
    check_convnet_capacity_one_dimensional_change2_spheres(128, RANGE_FACTORS(i), SAMPLES(i), 1, true, 1:154);
end

% Numericallay calculate capacity for manifolds with clipped axes
for D=10:10:50
    for id=1:154; check_convnet_capacity_one_dimensional_change2(128, 16.000, 201, 1, 0, id, nan, D, 1); end
    check_convnet_capacity_one_dimensional_change2(128, 16.000, 201, 1, 0, 1:154, nan, D, 1)
end

% Numerically calculate properties for manifolds with clipped axes
for D=10:10:50
    for id=1:154; check_convnet_covariance_low_rank_approx_optimal_K(128, 16.000, 201, 1, id, 5, 1, nan, 0, 1, 128, 0, 0, 0, D, 1); end
    check_convnet_covariance_low_rank_approx_optimal_K(128, 16.000, 201, 1, 1:154, 5, 1, nan, 0, 1, 128, 0, 0, 0, D, 1)
end

% Numericallay calculate capacity for manifolds with scaled center
for f=0.5:0.25:1.5
    for id=1:154; check_convnet_capacity_one_dimensional_change2(128, 16.000, 201, 1, 0, id, nan, 0, f); end
    check_convnet_capacity_one_dimensional_change2(128, 16.000, 201, 1, 0, 1:154, nan, 0, f)
end

% Numerically calculate properties for manifolds with scaled center
for f=0.5:0.25:1.5
    for id=1:154; check_convnet_covariance_low_rank_approx_optimal_K(128, 16.000, 201, 1, id, 5, 1, nan, 0, 1, 128, 0, 0, 0, 0, f); end
    check_convnet_covariance_low_rank_approx_optimal_K(128, 16.000, 201, 1, 1:154, 5, 1, nan, 0, 1, 128, 0, 0, 0, 0, f)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate capacity for different M-s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Numerically calculate capacity for alexnet using some of the samples
i=9; SAMPLE_JUMPS = [2,4,5,8,10,20,25,40,50,100,200,0];
for sample_jump=SAMPLE_JUMPS
    for run_id=1:7*22; check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, run_id, nan, nan, 1, sample_jump); end
    check_convnet_capacity_one_dimensional_change2(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 1:7*22, nan, nan, 1, sample_jump); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate properties using different N-s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=[128, 256, 512, 1024, 2048, 3072]
    for run_id=1:7*22; check_convnet_covariance_low_rank_approx_optimal_K(128, RANGE_FACTORS(i), SAMPLES(i), 1, run_id, 5, 1, nan, 0, 1, 128, 0, n); end
    check_convnet_covariance_low_rank_approx_optimal_K(128, RANGE_FACTORS(i), SAMPLES(i), 1, 1:7*22, 5, 1, nan, 0, 1, 128, 0, n); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate capacity for different P-s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(RANGE_FACTORS)
    % Numericallay calculate capacity for alexnet at P=128 (as above)
    for run_id=1:7*22; check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 0, '', run_id); end
    check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 0, '', 1:7*22);

    % Numericallay calculate capacity for alexnet at P=32
    for run_id=1:7*22; check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 0, '', run_id, 1, 0, false, 2, 32); end
    check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 0, '', 1:7*22, 1, 0, false, 2, 32);

    % Numericallay calculate capacity for alexnet at P=64
    for run_id=1:7*22; check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 0, '', run_id, 1, 0, false, 2, 64); end
    check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 0, '', 1:7*22, 1, 0, false, 2, 64);

    % Numericallay calculate capacity for alexnet at P=96
    for run_id=1:7*22; check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 0, '', run_id, 1, 0, false, 2, 96); end
    check_convnet_capacity_one_dimensional_change(128, RANGE_FACTORS(i), SAMPLES(i), 1, 0, 0, '', 1:7*22, 1, 0, false, 2, 96);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%     Randomly sampled manifolds data-set     %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RANGE_FACTORS = [0.125, 0.25, 0.5, 1, 2, 4, 8, 12, 16];
SAMPLES_2D = [51, 101, 201, 401, 801, 1601, 3201, 3201, 3201];
init_imagenet;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate capacity for 2d data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate capacity for high-dimensional data
for i=1:length(RANGE_FACTORS)
    % Numericallay calculate capacity for alexnet
    for run_id=1:2*22; check_convnet_capacity_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, 0, '', run_id); end
    check_convnet_capacity_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, 0, '', 1:2*22);
    
    % Numericallay calculate capacity with randomized centers for alexnet
    for run_id=1:2*22; check_convnet_capacity_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, 2, '', run_id); end
    check_convnet_capacity_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, 2, '', 1:2*22);

    % Numericallay calculate capacity with randomized manifolds for alexnet
    for run_id=1:2*22; check_convnet_capacity_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, 3, '', run_id); end   
    check_convnet_capacity_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, 3, '', 1:2*22);                          
    % Numericallay calculate capacity with randomized axes for alexnet
    for run_id=1:2*22; check_convnet_capacity_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, 5, '', run_id); end
    check_convnet_capacity_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, 5, '', 1:2*22);

    % Numericallay calculate capacity with permuted center and axes for alexnet
    for run_id=1:2*22; check_convnet_capacity_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, 7, '', run_id); end
    check_convnet_capacity_random_change(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 2, 7, '', 1:2*22);

    % Numericallay calculate capacity for vgg16
    for run_id=1:2*38; check_convnet_capacity_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, 0, '', run_id); end
    check_convnet_capacity_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 2, 0, '', 1:2*38);

    % Numericallay calculate capacity for resnet50
    for run_id=1:2*176; check_convnet_capacity_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 3, 2, 0, '', run_id); end
    check_convnet_capacity_random_change2(64, RANGE_FACTORS(i), SAMPLES_2D(i), 3, 2, 0, '', 1:2*176);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Theoretical capacity using low-rank approximation of correlations matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(RANGE_FACTORS)
    % Calculate low-rank properties for alexnet
    for run_id=1:2*22; check_convnet_covariance_low_rank_approx_optimal_K(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, run_id, 5, 2); end
    check_convnet_covariance_low_rank_approx_optimal_K(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 1:2*22, 5, 2);

    % Calculate low-rank properties for vgg16
    for run_id=1:2*38; check_convnet_covariance_low_rank_approx_optimal_K(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, run_id, 5, 2); end
    check_convnet_covariance_low_rank_approx_optimal_K(64, RANGE_FACTORS(i), SAMPLES_2D(i), 5, 1:2*38, 5, 2);

    % Calculate low-rank properties for resnet50
    for run_id=1:2*176; check_convnet_covariance_low_rank_approx_optimal_K(64, RANGE_FACTORS(i), SAMPLES_2D(i), 3, run_id, 5, 2); end
    check_convnet_covariance_low_rank_approx_optimal_K(64, RANGE_FACTORS(i), SAMPLES_2D(i), 3, 1:2*176, 5, 2);

    % Calculate capacity using low-rank approximation for alexnet before training
    i = 9; epoch=0;
    for run_id=1:2*21; check_convnet_covariance_low_rank_approx_optimal_K(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, run_id, 5, 2, epoch); end
    check_convnet_covariance_low_rank_approx_optimal_K(64, RANGE_FACTORS(i), SAMPLES_2D(i), 1, 1:2*21, 5, 2, epoch);
end