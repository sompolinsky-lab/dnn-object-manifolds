function [radius_results, mean_half_width1_results, mean_argmax_norm1_results, ...
    mean_half_width2_results, mean_argmax_norm2_results, effective_dimension_results, effective_dimension2_results, alphac_hat_results] = ...
    calc_manifolds_properties_fast(tuning_function, N_RANDOM_PROJECTIONS)
    if nargin < 2
        N_RANDOM_PROJECTIONS = 1000;
    end
    
    % A fast version of calc_manifolds_properties which calculate just radiuses
    [N_NEURONS, N_SAMPLES, N_OBJECTS] = size(tuning_function);

    % Reduce the global mean
    tuning_function = bsxfun(@minus, tuning_function, mean(nanmean(tuning_function, 2), 3));
    
    % Calculate centers
    Centers = reshape(nanmean(tuning_function, 2), [N_NEURONS, N_OBJECTS]);

    % Result variables
    center_norm_results = zeros(1, N_OBJECTS);
    object_mean_variance_results = zeros(1, N_OBJECTS);
    mean_half_width1_results = zeros(1, N_OBJECTS);
    mean_half_width2_results = zeros(1, N_OBJECTS);
    mean_argmax_norm1_results = zeros(1, N_OBJECTS);
    mean_argmax_norm2_results = zeros(1, N_OBJECTS);
    effective_dimension_results = zeros(1, N_OBJECTS);
    effective_dimension2_results = zeros(1, N_OBJECTS);
    alphac_hat_results = zeros(1, N_OBJECTS);

    for i=1:N_OBJECTS
        F = tuning_function(:,:,i);
        % Remove missing samples
        J = all(isfinite(F),1);
        F = F(:,J);

        % Substract the center
        cF = bsxfun(@minus, F, Centers(:,i));
        center_norm_results(i) = norm(Centers(:,i));

        % Calculate scaling
        n_samples = sum(J);
        minN = min(N_NEURONS, n_samples);
        scale_factor = sqrt(n_samples);

        % Lower dimensionality to D (if needed)
        [~, S, ~] = svd(cF, 0);
        si = diag(S)/scale_factor;
        assert(length(si) == minN);
        object_mean_variance_results(i) = norm(si(1:minN-1));

        [mean_half_width1, mean_argmax_norm1, mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat] = ...
            calc_manifold_properties(cF, center_norm_results(i), N_RANDOM_PROJECTIONS);
        mean_half_width1_results(i) = mean_half_width1;
        mean_argmax_norm1_results(i) = mean_argmax_norm1;
        mean_half_width2_results(i) = mean_half_width2;
        mean_argmax_norm2_results(i) = mean_argmax_norm2;
        effective_dimension_results(i) = effective_dimension;
        effective_dimension2_results(i) = effective_dimension2;
        alphac_hat_results(i) = alphac_hat;

        %assert_warn(abs(object_mean_variance_results(i).^2 - sum(sum(cF.^2, 1))scale_factor^2)<1e-3, ...
        %    sprintf('%1.3e %1.3e', object_mean_variance_results(i).^2, mean(sum(cF.^2, 1))));        
    end
    radius_results = object_mean_variance_results./center_norm_results;
end