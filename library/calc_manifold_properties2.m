function [mean_half_width1, mean_argmax_norm1, mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat, alphac_hat2, theory_center, ...
    TdS, dS_norm, Lambda, Delta, VF] = calc_manifold_properties2(tuning_function, center_norm, N_RANDOM_PROJECTIONS, kappa)
% Calculate manifold properties and predicted capacity using LS method,
% assuming no center-axis correlations
    function V = find_v(T, S, kappa, epsilon)
        % min ||v-t||^2 s.t. For all s in S v's+kappa<0 
        [n,d] = size(T);
        [d0,m] = size(S);
        assert(d == d0);
        
        C = eye(d);
        V = zeros(n,d);
        for i=1:n
            t = T(i,:);
            b = -ones(m,1)*kappa;
            v = calc_constrainted_least_square_cutting_plane(C, t', S', b, epsilon);
            % For larger problems this should be faster than: v = cplexlsqlin(C, t', S', b);
            V(i,:) = v;
        end
    end
    if nargin < 3
        N_RANDOM_PROJECTIONS = 1000;
    end
    if nargin < 4
        kappa = 0;
    end
    
    % Use only the needed amount of components, for efficient calculation
    assert(numel(center_norm) == 1);
    [N_NEURONS, N_SAMPLES] = size(tuning_function);

    minN = min(N_NEURONS, N_SAMPLES);
    D = minN - 1;
    
    % Substract the center
    original_center = mean(tuning_function, 2);
    cF = bsxfun(@minus, tuning_function, original_center);
    [orthogonal_projection, ~] = qr(cF, 0);
    orthogonal_projection = orthogonal_projection(:,1:D);
    F = [orthogonal_projection'*tuning_function; ones(1,N_SAMPLES)*center_norm];
    assert(all(size(F) == [minN, N_SAMPLES]));

    % Scale the data with the size of the centers
    TOLERANCE = 0.05/center_norm;
    F = F ./ center_norm;
    kappa = kappa ./ center_norm;

    % Calculate the values at the interia
    T = randn(N_RANDOM_PROJECTIONS, minN);
    t_norm = sqrt(sum(T.^2, 2));
    [~, argmax_margin] = max(T*F,[],2); 
    S = F(:,argmax_margin)';
    theory_center = mean(S,1);
    dS = bsxfun(@minus, S, theory_center);
    assert(all(size(dS) == [N_RANDOM_PROJECTIONS, minN]));
    dS_norm = sqrt(sum(dS.^2, 2)); 
    TdS=sum(T.*dS, 2);
    mean_half_width1 = mean(TdS);
    mean_argmax_norm1 = mean(dS_norm.^2);
    
    % If only the easy-to-calculate quantities are needed, bail out now
    if nargout <= 2
        return;
    end
    I = sum(T.*S,2)+kappa>=0;
    %fprintf('Found %d interia points\n', sum(~I));

    center = [zeros(D,1); 1];
    assert(all(size(center) == [minN, 1]));
    V = T;
    Lambda = nan(N_RANDOM_PROJECTIONS, 1);
    if sum(I)>0
        VI = find_v(T(I,:), F, kappa, TOLERANCE);
        V(I,:) = VI;
        lambda = (T(I,:)-VI)*center;
        Lambda(I,:) = lambda;
        S(I,:) = bsxfun(@rdivide, T(I,:)-VI, lambda);
        assert(max(abs(S(:,minN) - 1))<1e-10);
    end
    VF = V*F';
    Delta = sum((T-V).^2,2);
    
    theory_center = mean(S,1);
    dS = bsxfun(@minus, S, theory_center);
    assert(max(abs(dS(:,minN) - 0))<1e-10);
    dS_norm = sqrt(sum(dS.^2, 2));
    TdS=sum(T.*dS, 2);

    mean_half_width2 = mean(TdS);
    mean_argmax_norm2 = mean(dS_norm.^2);
    effective_dimension = mean(TdS.^2 ./ dS_norm.^2);
    effective_dimension2 = D*mean(TdS ./ (t_norm.*dS_norm)).^2;
    alphac_hat = 1./mean(1./theory_alpha0_cached(TdS-kappa)./(1+dS_norm.^2));
    alphac_hat2 = 1./mean(Delta);
    
    % Conver the theoretical center back into the original coordinates
    theory_center = theory_center .* center_norm;
    theory_center = orthogonal_projection * theory_center(1:D)';
    theory_center = theory_center + original_center;
end