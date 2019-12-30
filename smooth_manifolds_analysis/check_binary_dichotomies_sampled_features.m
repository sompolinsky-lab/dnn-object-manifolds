function [separability, features_used, labels_used] = check_binary_dichotomies_sampled_features(...
    XsAll, n, N_NEURON_SAMPLES, N_DICHOTOMIES, random_labeling_type, max_samples, global_preprocessing, verbose, features_type)
    if nargin < 9
        features_type = 0; % 0: sub-sample, 1: use the first n features (e.g. PCA), 2: random projections
    end

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
    labels_used = nan(N_NEURON_SAMPLES*N_DICHOTOMIES, N_OBJECTS);
    max_iterations = 1000;

    % Support for exhasive use of all dichotomies
    if N_DICHOTOMIES == 2^N_OBJECTS
        assert(N_NEURON_SAMPLES == 1, 'No point in sampling neurons when using all dichotomies');
        dichotomies=double(create_binary_dichotomies(N_OBJECTS));
        assert(all(size(dichotomies) == [N_DICHOTOMIES, N_OBJECTS]));
    end

    % Result variables
    current = 0;
    separability = zeros(N_NEURON_SAMPLES, N_DICHOTOMIES);
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
            if N_DICHOTOMIES == 2^N_OBJECTS
                y = dichotomies(i,:);
            elseif random_labeling_type == 2    % Sparse labeling
                y = ones(1,N_OBJECTS); 
                y(randi(N_OBJECTS, 1))=-1;
            elseif random_labeling_type == 1    % Balanced labeling
                y = ones(1,N_OBJECTS); 
                y(randperm(N_OBJECTS, round(N_OBJECTS/2)))=-1;
                assert(abs(sum(y)) <= 1, 'Non balanced labeling');
            else                                % IID labeling
                y = 2*randi([0, 1], [1,N_OBJECTS])-1;
            end
            labels_used(current, :) = y;
            Y = reshape(repmat(y, [N_SAMPLES, 1]), [1, N_SAMPLES*N_OBJECTS]);
            
            %[~, ~, margin]=check_linear_seperability_cplexlp(X, Y, 0, 1e-10);
            if max_samples>0
                [separable, ~, margin, samples_used]=check_linear_seperability_generalization_svm_cplexqp(Xs, y, 1e-10, false, max_iterations, max_samples);
                assert(isnan(margin) || separable == (margin > 0) || (samples_used == max_samples), sprintf('Margin and separability mismatch: %d %1.1e (used %d / %d samples)', separable, margin, samples_used, max_samples));
            else
                %[separable, ~, margin]=check_linear_seperability_svm_cplexqp_nobias(X, Y, 1e-10, false, max_iterations);
                [separable, ~, margin]=check_linear_seperability_svm_cplexqp(X, Y, 1e-10, false, max_iterations);
                assert(isnan(margin) || separable == (margin > 0), sprintf('Margin and separability mismatch: %d %1.1e', separable, margin));
            end
            separability(r, i) = margin > 0;

            if verbose
                fprintf('   N=%d margin=%1.1e (took %1.1f sec)\n', n, margin, toc(t));
            end
        end
    end
end
