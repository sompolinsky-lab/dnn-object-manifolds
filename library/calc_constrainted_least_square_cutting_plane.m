function [v,beta]=calc_constrainted_least_square_cutting_plane(C, t, S, b, TOLERANCE)
    % min 0.5*||Cv-t||^2 s.t. For all s in S v's<b 
    % The implementation is equivalent to:
    %                v = cplexlsqlin(C, t, S, b);
    % But using cutting-plane method to handle large problems
    % v     - the resulting solution
    % beta  - the Lagrange multiplier
    N = size(C, 1);
    M = size(S, 1);
    assert(all(size(C) == [N, N]));
    assert(all(size(t) == [N, 1]));
    assert(all(size(b) == [M, 1]));
    assert(all(size(S) == [M, N]));
    violation = inf;
    INITIAL_SAMPLES = min(5, M);
    MAX_SAMPLES = 1000;
    v = [];
    options = cplexoptimset();
    options.Display = 'off';

    name = hostname();
    % Behave well when running in ELSC cluster
    if isempty(name) || contains(name, 'brain') || contains(name, 'ielsc')
        options.threads = 1;
    end
 
    if nargin < 5
        TOLERANCE = 1e-3;
    end
    
    I = zeros(1, M);
    n_indices = INITIAL_SAMPLES;
    I(1:n_indices) = sample_indices(M, INITIAL_SAMPLES, 1);
    while violation > TOLERANCE && n_indices <= MAX_SAMPLES
        [v, ~, ~, flag, output, lambda] = cplexlsqlin(C, t, S(I(1:n_indices),:), b(I(1:n_indices)), [], [], [], [], options);
        %assert(sum(lambda.ineqlin>1e-8)==1);
        beta = lambda.ineqlin;
        if flag ~= 1
            disp(output)
            fprintf('Warning: Solution not found (flag=%d, %s, %s)\n', flag, output.message, output.cplexstatusstring);
            return;
        end
        %assert(norm(Cv-t) == resnorm);
        n_indices = n_indices + 1;
        [violation, i] = max(S*v-b);
        I(n_indices) = i;
    end
    if n_indices > MAX_SAMPLES
        fprintf('Warning: violation of %1.3f after %d iterations\n', violation, MAX_SAMPLES);
    end
end
