function [mean_half_width1, mean_argmax_norm1, mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat, alphac_hat2, ...
    theory_center, T, S, V, Lambda, Delta] = ...
    calc_manifold_properties3(tuning_function, center_norm, N_RANDOM_PROJECTIONS, kappa)
% Calculate manifold properties and predicted capacity using LS method
% without ignoring center-axis correlations.
    function [V, Lambda] = find_v(T, S, kappa, epsilon)
        % min ||v-t||^2 s.t. For all s in S v's+kappa<0 
        [n,d] = size(T);
        [d0,m] = size(S);
        assert(d == d0);
        
        C = eye(d);
        V = zeros(n,d);
        Lambda = zeros(n,1);
        for i=1:n
            t = T(i,:);
            b = -ones(m,1)*kappa;
            [v, beta] = calc_constrainted_least_square_cutting_plane(C, t', S', b, epsilon);
            % For larger problems this should be faster than: v = cplexlsqlin(C, t', S', b);
            V(i,:) = v;
            Lambda(i)= sum(beta);
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

    % Project to the effective embedding dimension to improve speed
    minN = min(N_NEURONS, N_SAMPLES);
    [orthogonal_projection, ~] = qr(tuning_function, 0);
    F = orthogonal_projection'*tuning_function;
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
    % Manifold's Steiner point
    theory_center = mean(S,1);
    theory_center_norm = norm(theory_center);
    dS = bsxfun(@minus, S, theory_center) ./ theory_center_norm;
    assert(all(size(dS) == [N_RANDOM_PROJECTIONS, minN]));
    dS_norm = sum(dS.^2, 2); 
    TdS=sum(T.*dS, 2);
    mean_half_width1 = mean(TdS);
    mean_argmax_norm1 = mean(dS_norm);
    
    % If only the easy-to-calculate quantities are needed, bail out now
    if nargout <= 2
        return;
    end
    I = sum(T.*S,2)+kappa>=0;
    %fprintf('Found %d interia points\n', sum(~I));

    V = T;
    Lambda = nan(N_RANDOM_PROJECTIONS, 1);
    if sum(I)>0
        [VI, lambda] = find_v(T(I,:), F, kappa, TOLERANCE);
        V(I,:) = VI;
        Lambda(I,:) = lambda;
        S(I,:) = bsxfun(@rdivide, T(I,:)-V(I,:), lambda);
    end
    Delta = sum((T-V).^2,2);
    
    %theory_center = mean(S,1);
    dS = bsxfun(@minus, S, theory_center);
    dS_norm = sqrt(sum(dS.^2, 2)); 
    S_norm = sqrt(sum(S.^2, 2)); 
    TdS = sum(T.*dS, 2); 
    TS = sum(T.*S, 2);
    mean_half_width2 = mean(TdS);
    mean_argmax_norm2 = mean(dS_norm.^2)./theory_center_norm^2;
    effective_dimension = mean(TdS.^2 ./ dS_norm.^2);
    effective_dimension2 = minN*mean(TdS ./ (t_norm.*dS_norm)).^2;
    alphac_hat = 1./mean(max(TS-kappa,0).^2 ./ S_norm.^2);
    alphac_hat2 = 1./mean(Delta);
    %assert(abs(alphac_hat2-alphac_hat) < 1e-6);
    
    % Conver the theoretical center back into the original coordinates
    theory_center =  (theory_center * center_norm)*orthogonal_projection';
    S = (S * center_norm) * orthogonal_projection';
    V = V * orthogonal_projection';
    T = T * orthogonal_projection';
end
