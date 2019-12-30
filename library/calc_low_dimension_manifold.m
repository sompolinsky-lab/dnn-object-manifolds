function result_tuning_function = calc_low_dimension_manifold(tuning_function, D, data_randomization, reduce_global_mean, radii_factor)
    if nargin < 3
        data_randomization = 0;
    end
    if nargin < 4
        reduce_global_mean = true;
    end
    if nargin < 5
        radii_factor = 1;       % Reduce the centers by this factor (default to 1, unchanged)
    end
    [N_NEURONS, N_SAMPLES, N_OBJECTS] = size(tuning_function);
    
    % Reduce the global mean
    if reduce_global_mean 
        tuning_function = bsxfun(@minus, tuning_function, mean(mean(tuning_function, 2), 3));
    end
    
    % Shuffle object assignment
    if data_randomization == 8
        shuffled_tuning = reshape(tuning_function, [N_NEURONS, N_SAMPLES*N_OBJECTS]);
        sampleIndices = randperm(N_SAMPLES*N_OBJECTS);
        shuffled_tuning = shuffled_tuning(:,sampleIndices);
        tuning_function =  reshape(shuffled_tuning, [N_NEURONS, N_SAMPLES, N_OBJECTS]);
    end
    
    % Calculate centers
    Centers = mean(tuning_function, 2);
	if data_randomization == 1
        ns=squeeze(sqrt(sum(Centers.^2,1)))';
        [Q, ~] = qr(squeeze(Centers));
        newCenters = reshape(bsxfun(@times, Q(:,1:N_OBJECTS), ns./radii_factor), size(Centers));
        ns2 = squeeze(sqrt(sum(newCenters.^2,1)))';
        assert(max(abs(ns2*radii_factor-ns)) < 1e-10);
    elseif data_randomization == 2 || data_randomization == 3
        ns=squeeze(sqrt(sum(Centers.^2,1)))';
        randCenters = randn(size(Centers));
        ns1=squeeze(sqrt(sum(randCenters.^2,1)))';
        newCenters = reshape(bsxfun(@times, squeeze(randCenters), ns./ns1./radii_factor), size(Centers));
        ns2 = squeeze(sqrt(sum(newCenters.^2,1)))';
        assert(max(abs(ns2*radii_factor-ns)) < 1e-8);
    elseif data_randomization == 4
        ns=squeeze(sqrt(sum(Centers.^2,1)))';
        randCenters = randn(size(Centers));
        ns1=squeeze(sqrt(sum(randCenters.^2,1)))';
        newCenters = reshape(bsxfun(@times, squeeze(randCenters), mean(ns)./ns1./radii_factor), size(Centers));
        ns2 = squeeze(sqrt(sum(newCenters.^2,1)))';
        assert(max(abs(ns2*radii_factor-mean(ns))) < 1e-8);
    else
        newCenters = Centers./radii_factor;
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
        if data_randomization == 3 || data_randomization == 4 || data_randomization == 5
            result_tuning_function(:,:,i) = bsxfun(@plus, dF(neuronIndices,:), newCenters(:,:,i));        
        elseif data_randomization == 7 
            result_tuning_function(:,:,i) = bsxfun(@plus, dF(neuronIndices,:), newCenters(neuronIndices,:,i));
        else
            result_tuning_function(:,:,i) = bsxfun(@plus, dF, newCenters(:,:,i));
        end
    end
end