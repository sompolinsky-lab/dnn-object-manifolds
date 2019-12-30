function [mean_half_width1, mean_argmax_norm1, mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat] = ...
    calc_manifold_properties(cF, center_norm, N_RANDOM_PROJECTIONS, kappa)
% Calculate manifold properties and predicted capacity using an iterative method
    if nargin < 3
        N_RANDOM_PROJECTIONS = 1000;
    end
    if nargin < 4
        kappa = 0;
    end
    TOLERANCE = 0.05/center_norm;
    %TOLERANCE=1e-2;
    MAX_ITER = 100;

    assert(center_norm>0, 'Norm must be possitive');
    N_NEURONS = size(cF, 1);
    
    % Scale the data with the size of the centers
    cF = cF ./ center_norm;
    %kappa0 = kappa;
    kappa = kappa ./ center_norm;
    w = randn(N_RANDOM_PROJECTIONS, N_NEURONS);
    w_norm = sqrt(sum(w.^2, 2));
    %w = sqrt(N_NEURONS)*bsxfun(@rdivide, w, w_norm); % Normalize w
    %random_projections_margin = mean(-min((w*cF),[],2)/2 + max((w*cF),[],2)/2);
    [max_random_projections_margin, argmax_random_projections_margin] = max(w*cF,[],2); 
    s0 = cF(:,argmax_random_projections_margin)';
    s0_norm_square = sum(s0.^2, 2); 
    ws0=sum(w.*s0, 2);
    assert_warn(max(abs(ws0-max_random_projections_margin))<1e-5, sprintf('%1.1e', max(abs(ws0-max_random_projections_margin))));
    mean_half_width1 = mean(ws0);
    mean_argmax_norm1 = mean(s0_norm_square);
    
    % If only the easy-to-calculate quantities are needed, bail out now
    if nargout <= 2
        return;
    end
    
    %original_ws0 = ws0;
    %original_s0_norm = s0_norm;
    %s0_norm_results = zeros(MAX_ITER, N_RANDOM_PROJECTIONS);
    %ws0_results = zeros(MAX_ITER, N_RANDOM_PROJECTIONS);

    %T=tic;
    eta = 0.1; % eta should be small 
    error = 1;
    iter = 0;
    error_results = zeros(MAX_ITER, N_RANDOM_PROJECTIONS);
    while error>TOLERANCE && iter < MAX_ITER
        iter = iter+1;
        %ws0_results(iter,:) = ws0;
        %s0_norm_results(iter,:) = s0_norm;
        z0=(ws0+kappa)./(1+s0_norm_square);
        dw=w-bsxfun(@times, z0,s0);
        [max_random_projections_margin, argmax_random_projections_margin] = max(dw*cF,[],2);
        error = max(max_random_projections_margin - sum(dw.*s0, 2));
        error_results(iter,:) = max_random_projections_margin - sum(dw.*s0, 2);
        if iter>1 && mean(error_results(iter,:)) > mean(error_results(iter-1,:))
            eta = 0.8*eta;
        end
        if error>TOLERANCE 
            s1 = cF(:,argmax_random_projections_margin)';
            s0 = (1-eta)*s0 + eta*s1;
            s0_norm_square = sum(s0.^2, 2);
            ws0=sum(w.*s0, 2);
        end
    end

    %if verbose
    %   fprintf('Kappa=%1.1f finished at iter #%d, error %1.3e (took %1.1f)\n', kappa0, iter, error, toc(T));
    %end

    %figure; plot(1:MAX_ITER, bsxfun(@rdivide, ws0_results(:,1:75), original_ws0(1:75)'))
    %figure; plot(1:MAX_ITER, bsxfun(@rdivide, s0_norm_results(:,1:75), original_s0_norm(1:75)'))
    %figure; plot(1:MAX_ITER, error_results');
    mean_half_width2 = mean(ws0);
    mean_argmax_norm2 = mean(s0_norm_square);
    effective_dimension = mean(ws0.^2 ./ s0_norm_square);
    effective_dimension2 = N_NEURONS*mean(ws0 ./ (w_norm.*sqrt(s0_norm_square))).^2;
    alphac_hat = 1./mean(1./theory_alpha0_cached(ws0+kappa)./(1+s0_norm_square));
end
