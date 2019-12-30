function [T, trials] = create_random_affine_transform_bounded(bound, degrees_of_freedom, param_id)
    % Sample random affine transformations until a bounded on is found
    trials = 0;
    current = bound+1;
    while current > bound
        trials = trials+1;
        % For DoF=6 2.4 means the cutoff is at the mean; for DoF=[2,4,6]
        % we found that the mean sigma satisfies sigma~0.286*DoF+0.684
        sigmaMean = 0.286*degrees_of_freedom+0.684;
        % Cutoff slightly above the mean
        T = create_random_affine_transform(bound/(sigmaMean*1.25), degrees_of_freedom, param_id); 
        current = calc_affine_transform_magnitude(T);
        %fprintf('%1.3f (bound=%1.3f)\n', current, bound);
    end
end