function result_tuning_function = calc_low_dimension_manifolds_preserving_correlations(tuning_function, Vk, D, center_norm_factor, local_preprocessing)
    [N_NEURONS, N_SAMPLES, N_OBJECTS] = size(tuning_function);
    if nargin < 3 || isnan(D)
        D = min(N_NEURONS, N_SAMPLES);  % Clip the PCA dimensions of each manifold to this valud
    end
    if nargin < 4
        center_norm_factor = 1;         % Scale the centers by this factor (default to 1, unchanged)
    end
    if nargin < 5
        local_preprocessing = 0; % 0- none, 1- orthogonalize centers, 2- random centers, 3- permuted axes and random centers, 5- permuted axes, 7- permute neurons
    end
    
    % Reduce the global mean
    global_mean = mean(mean(tuning_function, 2), 3);
    tuning_function = bsxfun(@minus, tuning_function, global_mean);
    % Reduce the common components
    common_components = reshape(Vk*(Vk'*reshape(tuning_function, [N_NEURONS, N_SAMPLES*N_OBJECTS])), size(tuning_function));
    tuning_function = bsxfun(@minus, tuning_function, common_components);
    
    % Shuffle object assignment
    if local_preprocessing == 8
        shuffled_tuning = reshape(tuning_function, [N_NEURONS, N_SAMPLES*N_OBJECTS]);
        sampleIndices = randperm(N_SAMPLES*N_OBJECTS);
        shuffled_tuning = shuffled_tuning(:,sampleIndices);
        tuning_function =  reshape(shuffled_tuning, [N_NEURONS, N_SAMPLES, N_OBJECTS]);
    end
    
    % Calculate centers
    Centers = mean(tuning_function, 2);
    ns=squeeze(sqrt(sum(Centers.^2,1)))';
	if local_preprocessing == 1
        [Q, ~] = qr(squeeze(Centers));
        newCenters = reshape(bsxfun(@times, Q(:,1:N_OBJECTS), ns.*center_norm_factor), size(Centers));
        ns2 = squeeze(sqrt(sum(newCenters.^2,1)))';
        assert(max(abs(ns2./center_norm_factor-ns)) < 1e-10);
	elseif local_preprocessing == 2 || local_preprocessing == 3        
        randCenters = randn(size(Centers));
        ns1=squeeze(sqrt(sum(randCenters.^2,1)))';
        newCenters = reshape(bsxfun(@times, squeeze(randCenters), ns./ns1.*center_norm_factor), size(Centers));
        ns2 = squeeze(sqrt(sum(newCenters.^2,1)))';
        assert(max(abs(ns2./center_norm_factor-ns)) < 1e-8);
	elseif local_preprocessing == 4
        randCenters = randn(size(Centers));
        ns1=squeeze(sqrt(sum(randCenters.^2,1)))';
        newCenters = reshape(bsxfun(@times, squeeze(randCenters), mean(ns)./ns1.*center_norm_factor), size(Centers));
        ns2 = squeeze(sqrt(sum(newCenters.^2,1)))';
        assert(max(abs(ns2./center_norm_factor-mean(ns))) < 1e-8);
	else
        newCenters = Centers.*center_norm_factor;
	end
    
    % Result variables
    result_tuning_function = zeros(size(tuning_function));
    for i=1:N_OBJECTS
        cF = bsxfun(@minus, tuning_function(:,:,i), Centers(:,:,i));
        % Remove missing samples
        J = all(isfinite(cF),1);
        cF = cF(:,J);

        [U, S, V] = svd(cF, 0);
        SD = zeros(size(S)); SD(1:D,1:D) = S(1:D,1:D);
        dF = U*SD*V';

        neuronIndices = randperm(N_NEURONS);
        if local_preprocessing == 3 || local_preprocessing == 4 || local_preprocessing == 5
            result_tuning_function(:,:,i) = bsxfun(@plus, dF(neuronIndices,:), newCenters(:,:,i));        
        elseif local_preprocessing == 7 
            result_tuning_function(:,:,i) = bsxfun(@plus, dF(neuronIndices,:), newCenters(neuronIndices,:,i));
        else
            result_tuning_function(:,:,i) = bsxfun(@plus, dF, newCenters(:,:,i));
        end
    end

    % Re-add the common components
    result_tuning_function = bsxfun(@plus, result_tuning_function, common_components);
    % Re-add the global mean
    result_tuning_function = bsxfun(@plus, result_tuning_function, global_mean);
end
