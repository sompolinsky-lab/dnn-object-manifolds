function [Vopt, Xopt, Kopt, residual_centers_norm, mean_square_corrcoef, mean_abs_corrcoef, mean_square_corr, mean_abs_corr] = ...
    optimal_low_rank_structure2(X, MAX_K, verbose, minSquare, N_REPEATS)
    % Problem dimensions
    [N, P] = size(X);
    
    % Default parameter values
    if nargin < 2
        MAX_K = ceil(P/2);
    end
    if nargin < 3
        verbose = 1;
    end
    if nargin < 4
        minSquare = true;   % T: minimize square correlations, F: minimize abs correlations
    end
    if nargin < 5
        N_REPEATS = 1;      % Number of optimization repeats, taking the most stabale one
    end
   
    early_termination = true;
    % Results variables
    mean_square_corr = nan(MAX_K+1,1);
    mean_abs_corr = nan(MAX_K+1,1);
    mean_square_corrcoef = nan(MAX_K+1,1);
    mean_abs_corrcoef = nan(MAX_K+1,1);
    residual_centers_norm = nan(MAX_K+1,P);

    % Add optimization package
    path_to_add = [pwd, '/../FOptM/'];
    if ~exist(path_to_add, 'dir')
        path_to_add = [pwd, '/FOptM/'];
    end
    path(path, path_to_add)

    % Optimization options
    MAX_ITER = 10000;
    opts = struct;
    opts.record = 0;    % no print out
    opts.mxitr  = MAX_ITER;  % max number of iterations
    opts.gtol = 1e-6;   % stop control for the projected gradient
    opts.xtol = 1e-6;   % stop control for ||X_k - X_{k-1}||
    opts.ftol = 1e-8;   % stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
                        % usually, max{xtol, gtol} > ftol

    % Move from representation from dimension of N to P-1, the rank
    if N > P-1
    [Q, ~] = qr(X, 0);
    assert(all(size(Q) == [N, P]));
    Q = Q(:,1:P-1);                 % [N, P-1]
    Xq = Q'*X;
    assert(all(size(Xq) == [P-1, P]));
    else
    Xq = X;
    Q = eye(N);
    end
    % Initialize cost to infinite
    bestCost = inf;
    Vopt = [];
    Kopt = 0;
    
    for ik=1:MAX_K+1
        k = ik - 1;
        if early_termination && (k > Kopt + 3)
            fprintf('Early termination Kopt=%d\n', Kopt);
            break;
        end
        tic;
        
        if k == 0
            V = [];
            Xk = Xq;
        else
            best_stability = 0;
            best_V = [];
            for in=1:N_REPEATS
                s = randn(P, 1);        % [P, 1]
                V0 = [Xq*s, V];        % [P-1, K]
                [V0, ~] = qr(V0, 0); assert(all(size(V0) == [size(Q,2), k]));
                [V1, output]= OptStiefelGBB(V0, @square_corrcoeff_full_cost, opts, Xq');
                assert_warn(output.itr < MAX_ITER, sprintf('Max iterations reached at k=%d (%s)', k, output.msg));
                assert_warn(isfinite(output.fval), sprintf('Non finite cost at k=%d: %1.3e (%s)', k, output.fval, output.msg));
                assert(output.itr == MAX_ITER || norm(V1'*V1-eye(k))< 1e-6, 'Deviation from I: %1.2e', norm(V1'*V1-eye(k)));
                cost_after = square_corrcoeff_full_cost(V1, Xq');
                assert(output.itr == MAX_ITER || ~isfinite(output.fval) || abs(output.fval - cost_after) < 1e-6, 'Cost differ: %1.3e <> %1.3e', output.fval, cost_after);
                Xk = Xq - V1*(V1'*Xq);  % [P-1, P]
                stability = min(sqrt(sum(Xk.^2, 1)) ./ sqrt(sum(Xq.^2, 1)));
                if stability > best_stability
                    best_stability = stability;
                    best_V = V1;
                end
                if N_REPEATS > 1 && verbose >= 2
                    fprintf(' [%d] cost=%1.3f stability=%1.3f\n', in, cost_after, stability);
                end
            end
            V = best_V;
            Xk = Xq - V*(V'*Xq);  % [P-1, P]
        end
        
        % Save output variables
        Xk_norm = sqrt(sum(Xk.^2, 1));
        residual_centers_norm(ik, :) = Xk_norm;
        Ck = Xk'*Xk;
        square_offdiagonal_corr = (Ck-diag(diag(Ck))).^2;
        mean_square_corr(ik) = sum(square_offdiagonal_corr(:))/(P-1)/P;
        abs_offdiagonal_corr = abs(Ck-diag(diag(Ck)));
        mean_abs_corr(ik) = sum(abs_offdiagonal_corr(:))/(P-1)/P;
        Ck0 = (Xk'*Xk) ./ (Xk_norm'*Xk_norm);
        square_offdiagonal_corr = (Ck0-diag(diag(Ck0))).^2;
        mean_square_offdiagonal_corr = sum(square_offdiagonal_corr(:))/(P-1)/P;
        mean_square_corrcoef(ik) = mean_square_offdiagonal_corr;
        abs_offdiagonal_corr = abs(Ck0-diag(diag(Ck0)));
        mean_abs_offdiagonal_corr = sum(abs_offdiagonal_corr(:))/(P-1)/P;
        mean_abs_corrcoef(ik) = mean_abs_offdiagonal_corr;
        if verbose >= 1
            fprintf('k=%d <square>=%1.4f <abs>=%1.3f (took %1.1f sec)\n', k, mean_square_offdiagonal_corr, mean_abs_offdiagonal_corr, toc);
        end
        
        % Update the best results
        if minSquare
            currentCost = mean_square_offdiagonal_corr;
        else
            currentCost = mean_abs_offdiagonal_corr;
        end
        if currentCost < bestCost
            bestCost = currentCost;
            if isempty(V)
                Vopt = V;
            else
                Vopt = Q*V;
            end
            Xopt = Q*Xk;
            Kopt = k;
        end
    end
end

