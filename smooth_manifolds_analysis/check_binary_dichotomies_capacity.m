function [Nc, separablity_results, Ns, radius_results, mean_half_width1_results, mean_argmax_norm1_results, ...
    mean_half_width2_results, mean_argmax_norm2_results, effective_dimension_results, effective_dimension2_results, ...
    alphac_hat_results, features_used_results, labels_used_results] = ...
check_binary_dichotomies_capacity(XsAll, N_NEURON_SAMPLES, N_DICHOTOMIES, verbose, random_labeling_type, ...
    precision, max_samples, global_preprocessing, features_type, jumps, properties_type)
    % Calculate the capacity, the minimal N for which over half of binary
    % dichotomies are achievable. Work in steps of 128, then do binary search.
    if nargin < 2
        N_NEURON_SAMPLES = 41; % an even number to prevent a 50% result
    end
    if nargin < 3
        N_DICHOTOMIES = 1;     % number of repeats per n
    end
    if nargin < 4
        verbose=false;
    end
    if nargin < 5
        random_labeling_type=0; % default to IID
    end
    if nargin < 6
        precision=1;
    end
    if nargin < 7
        max_samples = 0;
    end
    if nargin < 8
        global_preprocessing = 0;
    end
    if nargin < 9
        features_type = 0; % 0: sub-sample, 1: use the first n features (e.g. PCA), 2: random projections
    end
    % Get problem dimensions
    assert(length(size(XsAll)) == 3, 'Data must be [N_NEURONS, N_SAMPLES, N_OBJECTS]');
    [N_NEURONS, N_SAMPLES, N_OBJECTS] = size(XsAll);    
    if nargin < 10 || isempty(jumps)
        if N_NEURONS < 100
            jumps = 10;
        elseif mod(N_NEURONS, 100) == 0
            jumps = 100;
        else
            jumps = 128;
        end
    end
    if nargin < 11
        properties_type = 1; % 0: none, 1: old iterative theory, 2: new LS theory
    end

    if verbose
        fprintf(' %d neurons %d conditions %d objects\n', N_NEURONS, N_SAMPLES, N_OBJECTS);
    end

    % Result variable
    separablity_results = nan(N_NEURONS, 1);
    if N_NEURONS <= 4096 && features_type<2
        features_used_results = nan(N_NEURONS, N_NEURON_SAMPLES, N_NEURONS);
    else
        features_used_results = [];
    end
    labels_used_results = nan(N_NEURONS, N_DICHOTOMIES*N_NEURON_SAMPLES, N_OBJECTS);
    
    % Find max N in jumps 
    for n=jumps:jumps:(N_NEURONS-mod(N_NEURONS,jumps))
        if verbose
            fprintf('  initial check at n=%d\n', n);
            T=tic;
        end
        [separability, features_used, labels_used] = check_binary_dichotomies_sampled_features(XsAll, n, N_NEURON_SAMPLES, ...
            N_DICHOTOMIES, random_labeling_type, max_samples, global_preprocessing, verbose>1, features_type);
        separablity_results(n) = mean(separability(:));
        if size(features_used_results,1) > n
            features_used_results(n,:,1:n) = features_used;
        end
        labels_used_results(n,:,:) = labels_used;

        if verbose
            fprintf('  N=%d %1.2f separable (took %1.1f sec)\n', n, mean(separability(:)), toc(T));
        end
        if separablity_results(n) == 1
            if verbose
                fprintf('  Found initial max N\n');
            end
            break;
        end
    end
    
    % Binary search until precision is met
    max_N = n;
    min_N = find(separablity_results == 0, 1, 'last');
    if isempty(min_N)
        min_N = 0;
    end
    [min_value, min_index] = min(separablity_results);
    [max_value, max_index] = max(separablity_results);
    while (max_N - min_N > precision) || (min_value>0 && min_index-1>precision) || (max_value<1 && N_NEURONS-max_index>precision)
        if (max_N - min_N > precision)
            n = ceil((max_N + min_N)/2);
        elseif (min_value>0 && min_index-1>precision && ceil((min_index+1)/2)~=n)
            n = ceil((min_index+1)/2);
        elseif (max_value<1 && N_NEURONS-max_index>precision && ceil((max_index+N_NEURONS)/2)~=n)
            n = ceil((max_index+N_NEURONS)/2);
        else
            break;
        end
        if verbose
            fprintf('  looking at n=%d [%d-%d] (%d %d %d)\n', n, min_N, max_N, (max_N - min_N > precision), ...
                min(separablity_results)>0, max(separablity_results)<1);
            T=tic;
        end
        
        [separability, features_used, labels_used] = check_binary_dichotomies_sampled_features(XsAll, n, N_NEURON_SAMPLES, ...
            N_DICHOTOMIES, random_labeling_type, max_samples, global_preprocessing, verbose>1, features_type);
        separablity_results(n) = mean(separability(:));
        if size(features_used_results,1) > n
            features_used_results(n,:,1:n) = features_used;
        end
        labels_used_results(n,:,:) = labels_used;

        if verbose
            fprintf('  N=%d %1.2f separable (took %1.1f sec)\n', n, mean(separability(:)), toc(T));
        end
        [min_value, min_index] = min(separablity_results);
        [max_value, max_index] = max(separablity_results);
        if (max_N - min_N > precision)
            if separablity_results(n) > 0.5
                max_N = n;
            else
                min_N = n;
            end
        end
    end
    %Nc = find(separablity_results <= 0.5, 1, 'last');
    Nc = find(separablity_results >= 0.5, 1);
    capacity = N_OBJECTS / Nc;
    Ns = find(isfinite(separablity_results));
    if ~isempty(features_used_results)
        features_used_results = reshape(squeeze(features_used_results(Nc, :, 1:Nc)), [N_NEURON_SAMPLES, Nc]);
    end
    labels_used_results = squeeze(labels_used_results(Nc, :, :));
    
    radius_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    mean_half_width1_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    mean_argmax_norm1_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    mean_half_width2_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    mean_argmax_norm2_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    effective_dimension_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    effective_dimension2_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    alphac_hat_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    
    if properties_type == 0
        if verbose
            fprintf(' Critical N=%d alpha=%1.2f\n', Nc, capacity);
        end
        return;
    end
    for r=1:N_NEURON_SAMPLES
        % Prepare features
        if isempty(features_used_results)
             random_projections = randn(N_NEURONS, Nc)./sqrt(N_NEURONS);
             X = random_projections'*reshape(XsAll, [N_NEURONS, N_SAMPLES*N_OBJECTS]);
             Xs = reshape(X, [Nc, N_SAMPLES, N_OBJECTS]);     
        else
             I = features_used_results(r,:);
             Xs = XsAll(I,:,:);
        end
        
        % Calculate properties
        [current_radius, current_half_width1, current_argmax_norm1, current_half_width2, current_argmax_norm2, ...
            current_effective_dimension, current_effective_dimension2, current_alphac_hat] = calc_manifolds_properties_fast(Xs);
        radius_results(r,:) = current_radius;
        mean_half_width1_results(r,:) = current_half_width1;
        mean_argmax_norm1_results(r,:) = current_argmax_norm1;
        mean_half_width2_results(r,:) = current_half_width2;
        mean_argmax_norm2_results(r,:) = current_argmax_norm2;
        effective_dimension_results(r,:) = current_effective_dimension;
        effective_dimension2_results(r,:) = current_effective_dimension2;
        alphac_hat_results(r,:) = current_alphac_hat;
    end

    if verbose
        alpha_hat = mean(1./mean(1./alphac_hat_results, 2), 1);
        fprintf(' Critical N=%d alpha=%1.2f alpha_hat=%1.2f (Rv=%1.2f Rhw=%1.2f)\n', Nc, capacity, alpha_hat, mean(radius_results(:)), mean(mean_half_width1_results(:)));
    end
end

