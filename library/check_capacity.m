function check_capacity(name_prefix, session_ids, data_randomization, data_preprocessing, expected_dimension)
% Calculate capacity for the given manifolds.
% This version uses an adaptive number of samples and does not collect manifold properties 
% to minimize computational cost.
if nargin < 2
    session_ids = 0;
end
if nargin < 3
    data_randomization = 0; % 0- none, 1- orthogonalize centers, 2- random centers, 3- permuted manifold
end
if nargin < 4
    data_preprocessing = 0; % 0- none, 1- znorm, 2- whitening, 3- centers decorelation
end
if nargin < 5
    expected_dimension = [];
end

% Parameters

features_type = 2; % 0: sub-sample, 1: use the first n features (e.g. PCA), 2: random projections
random_labeling_type = 1; % 0- binary iid, 1- balanced, 2- sparse

% Expected precision ep=sqrt(pq/n) so that for p=q=0.5 ep=0.05 yields n=100
EXPECTED_PRECISION = 0.05; 
max_samples = 0;
precision = 1;

suffix = '';
if random_labeling_type == 1
    suffix = [suffix, '_balanced'];
elseif random_labeling_type == 2
    suffix = [suffix, '_sparse'];
end
if data_preprocessing == 1
    suffix = [suffix, '_znorm'];
elseif data_preprocessing == 2
    suffix = [suffix, '_whiten'];
elseif data_preprocessing == 3
    suffix = [suffix, '_centers_whiten'];
end
if data_randomization == 0
elseif data_randomization == 1
    suffix = [suffix, '_orth'];
elseif data_randomization == 2
    suffix = [suffix, '_centers_random'];
elseif data_randomization == 3
    suffix = [suffix, '_manifold_random'];
elseif data_randomization == 4
    suffix = [suffix, '_manifold_random_uniform_centers'];
elseif data_randomization == 5
    suffix = [suffix, '_axes_random'];
elseif data_randomization == 7
    suffix = [suffix, '_permute_random'];
else
    assert(data_randomization == 8);
    suffix = [suffix, '_shuffle'];
end
if length(session_ids) == 1 && session_ids>0
    suffix = sprintf('%s_s%d', suffix, session_ids);
end
in_name = sprintf('%s_tuning.mat', name_prefix);
out_name = sprintf('%s_capacity%s.mat', name_prefix, suffix);
assert(exist(in_name, 'file')>0, 'File not found: %s', in_name);

in_file = matfile(in_name);
tuning_size = size(in_file, 'tuning_function');
if length(tuning_size) == 4
    [N_SESSIONS, N_NEURONS, N_SAMPLES, N_OBJECTS] = size(in_file, 'tuning_function');
else
    assert(length(tuning_size) == 3, 'Tuning function is not of size 3,4');
    N_SESSIONS = 1;
    [N_NEURONS, N_SAMPLES, N_OBJECTS] = size(in_file, 'tuning_function');
end

if ~isempty(expected_dimension)
    assert(isnan(expected_dimension(1)) || expected_dimension(1) == N_SESSIONS, 'Expected %d sessions, found %d', expected_dimension(1), N_SESSIONS);
    assert(isnan(expected_dimension(2)) || expected_dimension(2) == N_OBJECTS, 'Expected %d objects, found %d', expected_dimension(2), N_OBJECTS);
    assert(isnan(expected_dimension(3)) || expected_dimension(3) == N_SAMPLES, 'Expected %d samples, found %d', expected_dimension(3), N_SAMPLES);
    assert(isnan(expected_dimension(4)) || expected_dimension(4) == N_NEURONS, 'Expected %d neurons, found %d', expected_dimension(4), N_NEURONS);
end
assert(all(session_ids <= N_SESSIONS), 'session_ids must be smaller than N_SESSIONS');

% Result variables
capacity_results            = nan(N_SESSIONS, 1);
separability_results        = nan(N_SESSIONS, N_NEURONS);
neuron_samples_used_results = nan(N_SESSIONS, N_NEURONS);

fprintf('Results saved to %s\n', out_name);
if exist(out_name, 'file')
    fprintf('Loading existing results\n');
    load(out_name);
end

if length(session_ids)>1
    fprintf('Collecting previousy generated results\n');
    for s=session_ids
        run_name = sprintf('%s_capacity%s_s%d.mat', name_prefix, suffix, s);
        assert(exist(run_name, 'file')>0, 'Missing file: %s', run_name);
        run_file = matfile(run_name);

        capacity_results(s,:) = run_file.capacity_results(s,:);
        separability_results(s,:) = run_file.separability_results(s,:);
        neuron_samples_used_results(s,:) = run_file.neuron_samples_used_results(s,:);
    end
    
    save(out_name, 'capacity_results', 'separability_results', 'neuron_samples_used_results', ...
        'EXPECTED_PRECISION', 'max_samples', 'precision', 'features_type', 'random_labeling_type', '-v7.3');
    return;
end

if session_ids == 0
    session_ids = 1:N_SESSIONS;
end

% Get session titles
if isprop(in_file, 'data_titles')
    data_titles = in_file.data_titles;
else
    data_titles = cell(N_SESSIONS,1);
    for s=1:N_SESSIONS
        data_titles{s} = sprintf('session #%d', s);
    end
end

for s = session_ids
    data_title = data_titles{s};
    capacity = capacity_results(s);
    if any(isfinite(separability_results(s,:)))
        fprintf('Skipping existing session %s (Ac=%1.2f)\n', data_title, N_OBJECTS./capacity);
        continue;
    end
    fprintf('Working on %s\n', data_title);
    T=tic;

    tic;
    if length(tuning_size) == 3
        full_tuning_function = in_file.tuning_function;
    else
        full_tuning_function = squeeze(in_file.tuning_function(s,:,:,:));
    end
    assert(all(size(full_tuning_function) == [N_NEURONS, N_SAMPLES, N_OBJECTS]));
    %mean_squeare_sample_response = mean(mean(full_tuning_function.^2, 3), 1); 
    %assert(length(mean_squeare_sample_response) == N_SAMPLES);
    %nzSamples = find(isfinite(mean_squeare_sample_response));
    %N_SAMPLES = length(nzSamples);
    %full_tuning_function = double(full_tuning_function(:,nzSamples,:));
    mean_squeare_firing_rate = nanmean(mean(full_tuning_function.^2, 3), 2);
    nzIndices = find(mean_squeare_firing_rate > 0);
    N = length(nzIndices);
    fprintf('Loaded %s data [N=%d M=%d P=%d] (took %1.1f sec)\n', data_title, N, N_SAMPLES, N_OBJECTS, toc);

    % Perform randomization if needed
    if data_randomization == 0
        current_tuning_function = double(full_tuning_function(nzIndices,:,:));
    else
        minN = min(N, N_SAMPLES);
        current_tuning_function = calc_low_dimension_manifold(double(full_tuning_function(nzIndices,:,:)), minN, data_randomization);
    end
    % Perform pre-processing if needed
    if data_preprocessing == 1
        neuron_mean = mean(current_tuning_function(:,:),2);
        neuron_std = std(current_tuning_function(:,:),[],2);
        current_tuning_function = (current_tuning_function - neuron_mean) ./ neuron_std;
    end

    [capacity, separability, ~, n_neuron_samples_used] = ...
        check_binary_dichotomies_capacity2(current_tuning_function, EXPECTED_PRECISION, true, random_labeling_type, ...
            precision, max_samples, 0, features_type);

    capacity_results(s) = capacity;
    separability_results(s, 1:N) = separability;
    neuron_samples_used_results(s, 1:N) = n_neuron_samples_used;

    save(out_name, 'capacity_results', 'separability_results', 'neuron_samples_used_results', ...
        'EXPECTED_PRECISION', 'max_samples', 'precision', 'features_type', 'random_labeling_type', '-v7.3');
    fprintf('Done. (took %1.1f hours, Ac=%1.2f)\n', toc(T)/3600., N_OBJECTS./capacity);
end
end