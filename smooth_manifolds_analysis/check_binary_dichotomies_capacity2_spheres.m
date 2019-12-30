function [Nc, separablity_results, Ns, n_neuron_samples_used, n_support_vectors] = check_binary_dichotomies_capacity2_spheres(...
    centers, axes, EXPECTED_PRECISION, verbose, random_labeling_type, precision, max_samples, features_type, jumps)
% Calculate the capacity, the minimal N for which over half of binary dichotomies are achievable. 
% Work in jumps to find an upper bound, then do binary search. 
%
% This version uses an adaptive number of samples and does not collect manifold properties 
% to minimize computational cost.
% 
% This version works only on spheres.
%
    if nargin < 3
        EXPECTED_PRECISION = 0.05;
    end
    if nargin < 4
        verbose=2;
    end
    if nargin < 5
        random_labeling_type=0; % default to IID
    end
    if nargin < 6
        precision = 1;
    end
    if nargin < 7
        max_samples = 0;
    end
    if nargin < 8
        features_type = 0; % 0: sub-sample, 1: use the first n features (e.g. PCA), 2: random projections
    end
    % Get problem dimensions
    assert(length(size(centers)) == 2, 'Centers must be [N_NEURONS, N_OBJECTS]');
    assert(length(size(axes)) == 3, 'Centers must be [N_NEURONS, N_AXES, N_OBJECTS]');
    [N_NEURONS, N_AXES, N_OBJECTS] = size(axes);
    assert(all(size(centers) == [N_NEURONS, N_OBJECTS]));
    
    if nargin < 9 || isempty(jumps)
        if N_NEURONS < 100
            jumps = 10;
        elseif N_NEURONS < 1024
            if mod(N_NEURONS, 100) == 0
                jumps = 100;
            else
                jumps = 128;
            end
        else
            if mod(N_NEURONS, 500) == 0
                jumps = 500;
            else
                jumps = 512;
            end
        end
    end

    if verbose
        fprintf(' %d neurons %d axes %d objects\n', N_NEURONS, N_AXES, N_OBJECTS);
    end

    % Result variable
    separablity_results = nan(N_NEURONS, 1);
    n_neuron_samples_used = nan(N_NEURONS, 1);
    n_support_vectors = nan(N_NEURONS, N_OBJECTS);
    
    % Find max N in jumps 
    J = unique([jumps:jumps:(N_NEURONS-mod(N_NEURONS,jumps)), N_NEURONS]);
    for n=J
        if verbose
            fprintf('  initial check at n=%d\n', n);
            T=tic;
        end
        if isnan(separablity_results(n))
            [separability, nsv_per_object] = check_binary_dichotomies_sampled_features2_spheres(centers, axes, n, EXPECTED_PRECISION, ...
                random_labeling_type, max_samples, verbose>1, features_type);
            n_neuron_samples_used(n) = sum(isfinite(separability(:)));
            separablity_results(n) = nanmean(separability(:));
            n_support_vectors(n,:) = nsv_per_object;
        end
        if verbose
            fprintf('  N=%d %1.2f separable (%d samples, took %1.1f sec)\n', n, separablity_results(n), n_neuron_samples_used(n), toc(T));
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
    n1 = nan;
    n2 = nan;
    while (max_N - min_N > precision) || (min_value>0 && min_index-1>precision && ceil((min_index+1)/2)~=n) || (max_value<1 && N_NEURONS-max_index>precision && ceil((max_index+N_NEURONS)/2)~=n)
        if (max_N - min_N > precision)
            n = ceil((max_N + min_N)/2);
        elseif (min_value>0 && min_index-1>precision && ceil((min_index+1)/2)~=n)
            n = ceil((min_index+1)/2);
        elseif (max_value<1 && N_NEURONS-max_index>precision && ceil((max_index+N_NEURONS)/2)~=n)
            n = ceil((max_index+N_NEURONS)/2);
        else
            break;
        end
        if n == n1 || n == n2
            break;
        end
        if verbose
            fprintf('  looking at n=%d [%d-%d] (%d %d %d)\n', n, min_N, max_N, (max_N - min_N > precision), ...
                min(separablity_results)>0, max(separablity_results)<1);
            T=tic;
        end
        
        if isnan(separablity_results(n))
            [separability, nsv_per_object] = check_binary_dichotomies_sampled_features2_spheres(centers, axes, n, EXPECTED_PRECISION, ...
                random_labeling_type, max_samples, verbose>1, features_type);
            n_neuron_samples_used(n) = sum(isfinite(separability(:)));
            separablity_results(n) = nanmean(separability(:));
            n_support_vectors(n,:) = nsv_per_object;
        end

        if verbose
            fprintf('  N=%d %1.2f separable (%d samples, took %1.1f sec)\n', n, separablity_results(n), n_neuron_samples_used(n), toc(T));
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
        n2 = n1;
        n1 = n;
    end
    %Nc = find(separablity_results <= 0.5, 1, 'last');
    Nc = find(separablity_results >= 0.5, 1);
    if isempty(Nc)
        Nc = nan;
        if verbose; fprintf(' Critical N not found\n'); end
    else
        if verbose; fprintf(' Critical N=%d alpha=%1.2f\n', Nc, N_OBJECTS / Nc); end
    end
    Ns = find(isfinite(separablity_results));        
end

