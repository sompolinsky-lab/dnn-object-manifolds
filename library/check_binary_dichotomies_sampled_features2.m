function [separability, nsv_per_cluster] = check_binary_dichotomies_sampled_features2(...
    XsAll, n, expected_precision, random_labeling_type, max_samples, global_preprocessing, verbose, features_type)
    if nargin < 8
        features_type = 0; % 0: sub-sample, 1: use the first n features (e.g. PCA), 2: random projections
    end
    % The minimal variance is when the dichotomies are independently sampled
    N_DICHOTOMIES = 1;
    % In order to reach the precision we need to try a minimal number of them
    MIN_NEURON_SAMPLES = ceil(1/expected_precision);
    % Expected precision ep=sqrt(pq/n) so that for p=q=0.5 n=(0.5/ep)^2
    N_NEURON_SAMPLES = ceil((0.5/expected_precision)^2);
    % Check linear separability for the given number of binary dichotomies
    % using n sub-sampled features.
    [N_NEURONS, N_SAMPLES, N_OBJECTS] = size(XsAll);
    if features_type == 0
        features_used = sample_indices(N_NEURONS, n, N_NEURON_SAMPLES);
    elseif features_type == 1
        assert(N_NEURON_SAMPLES == 1);
        features_used = zeros(1, n); features_used(1:n) = 1:n;
    else
        assert(features_type == 2);
        features_used = [];
    end
    %labels_used = nan(N_NEURON_SAMPLES*N_DICHOTOMIES, N_OBJECTS);
    max_iterations = 1000;
    tolerance = 1e-10;
    
    % Result variables
    current = 0;
    separability = nan(N_NEURON_SAMPLES, N_DICHOTOMIES);
    for r=1:N_NEURON_SAMPLES
        % Prepare features
        if isempty(features_used)
            random_projections = randn(N_NEURONS, n)./sqrt(N_NEURONS);
            X = random_projections'*reshape(XsAll, [N_NEURONS, N_SAMPLES*N_OBJECTS]);
            Xs = calc_randomization_single_neurons(reshape(X, [n, N_SAMPLES, N_OBJECTS]), global_preprocessing);
        else
            I = features_used(r,:);
            Xs = calc_randomization_single_neurons(XsAll(I,:,:), global_preprocessing);
        end
        X = reshape(Xs, [n, N_SAMPLES*N_OBJECTS]);
    
        % Calculate if we are above or below capacity
        for i=1:N_DICHOTOMIES
            current = current + 1;
            t=tic;
            y=sample_random_labels(N_OBJECTS, random_labeling_type);
            
            %labels_used(current, :) = y;
            Y = reshape(repmat(y, [N_SAMPLES, 1]), [1, N_SAMPLES*N_OBJECTS]);
            
            %[~, ~, margin]=check_linear_seperability_cplexlp(X, Y, 0, tolerance);
            if max_samples>0
                [separable, ~, margin, samples_used, ~, sv_indices]=check_linear_seperability_generalization_svm_cplexqp(Xs, y, tolerance, false, max_iterations, max_samples);
                assert(isnan(margin) || separable == (margin > 0) || (samples_used == max_samples), sprintf('Margin and separability mismatch: %d %1.1e (used %d / %d samples)', separable, margin, samples_used, max_samples));
            else
                %[separable, ~, margin]=check_linear_seperability_svm_cplexqp_nobias(X, Y, tolerance, false, max_iterations);
                [separable, ~, margin, ~, ~, sv_indices]=check_linear_seperability_svm_cplexqp(X, Y, tolerance, false, max_iterations);
                assert(isnan(margin) || separable == (margin > 0), sprintf('Margin and separability mismatch: %d %1.1e', separable, margin));
            end
            nsv_per_cluster = zeros(N_SAMPLES, N_OBJECTS); nsv_per_cluster(sv_indices) = 1; nsv_per_cluster = sum(nsv_per_cluster, 1);
            separability(r, i) = margin > 0;

            if verbose
                fprintf('   N=%d margin=%1.1e (took %1.1f sec)\n', n, margin, toc(t));
            end
        end
        nRes = sum(isfinite(separability(:)));
        p = nansum(separability(:) == 1)/nRes;
        q = nansum(separability(:) == 0)/nRes;
        assert(p+q == 1);
        std_of_the_mean = sqrt(p*q/nRes);
        if nRes>=MIN_NEURON_SAMPLES && std_of_the_mean <= expected_precision
            if verbose
                fprintf('   reached target std of the mean: %1.3f\n', std_of_the_mean);
            end
            return;
        end
    end
end
