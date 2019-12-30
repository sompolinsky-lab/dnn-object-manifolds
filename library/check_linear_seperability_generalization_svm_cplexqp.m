%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check for an input data-set of vectors X and labeling y if they are
% linearly seperable by a plane (with bias). 
%
% X is [NxMxP] and represents M samples from P clusters, each of size N. 
% y is [1xP] and represents the cluster labeling.
%
% This method finds the max-margin linear seperation using either primal
% or dual formalism.
%
% To handle generalization with large problems, a small number of
% conditions (indicated by 'initial_conditions') is used to find a
% separating plane, and then the sample with the worst field is added, 
% iterating until all points are classified correctly or the problem 
% is unseparable (no slack variables are used here).
%
% Note this implementation adds the worst item per object, not globally.
% Thus samples_used is samples used per object and max_samples is max
% samples per object (unlike the version with slack variables).
%
% Author: Uri Cohen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [separable, best_w, margin, samples_used, nsv, sv_indices, lagrange_multipliers]=check_linear_seperability_generalization_svm_cplexqp(Xs, y, tolerance, solve_dual, max_iterations, max_samples, initial_conditions, verbose)
    % Assert input dimensions
    M = size(Xs, 2);    % Samples per cluster
    P = size(Xs, 3);    % Number of clusters
    N = size(Xs, 1);    % Data size
    assert(all(size(Xs) == [N, M, P]), 'x must be NxMxP');
    assert(all(size(y) == [1, P]), 'y must be P labels');
    assert(all(abs(y) == 1), 'y must be +1/-1');
    a = all(isfinite(Xs),1);
    assert(all(a(:)), 'Objects with emtpy samples are not currently supported');
    
    if nargin < 3
        tolerance = 1e-10;
    end
    if nargin < 4
        solve_dual = false;
    end
    if nargin < 5
        max_iterations = 0;
    end
    if nargin < 6
        max_samples = M;
    end
    if nargin < 7
        initial_conditions = min(5, M);
    end
    if nargin < 8
        verbose = false;
    end
    
    % Data with bias
    Xb = [Xs; ones(1, M, P)];
    assert(all(size(Xb) == [N+1, M, P]), 'x must be NxMxP');
    
    % Find indices of initial conditions
    I = sample_indices(M, initial_conditions, P); % [P x K]
    assert(all(size(I) == [P, initial_conditions]));
    
    % Look for separability with current indices until data is not
    % separable or problem limits are encountered.
    separable = true;
    test_margin = 0;
    train_margin = 1;
    K = size(I,2);
    while separable && test_margin < 0.99*train_margin && K<=max_samples
        currXs = zeros(N, K, P);
        for i=1:P
            currXs(:,:,i) = Xs(:,I(i,:),i);
        end
        currX = reshape(currXs, [N, K*P]);
        currY = reshape(repmat(y, [K, 1]), [1, K*P]);
        samples_used = K;
        [separable, best_w, train_margin, ~, nsv, sv_indices, lagrange_multipliers_samples] = ...
        check_linear_seperability_svm_cplexqp(currX, currY, tolerance, solve_dual, max_iterations);
        if numel(lagrange_multipliers_samples) == 0
            lagrange_multipliers=[];
        else
            lagrange_multipliers_samples = reshape(lagrange_multipliers_samples, [K, P]);
                lagrange_multipliers = zeros(M, P); 
            for i=1:P
                    lagrange_multipliers(I(i,:),i) = lagrange_multipliers_samples(:,i);
            end
        end    
        if separable
            I = [I, zeros(P,1)];
            K = size(I,2);
            
            wnorm = norm(best_w(1:N), 2);
            test_margin = train_margin;
            % Add the worst sample of each cluster
            %TODO: calculate in matrix form? min(best_w'*Xb.*y) / wnorm;
            for i=1:P
                h = y(i)*best_w'*Xb(:,:,i);
                [h0, i0] = min(h./wnorm);
                test_margin = min(test_margin, h0);
                I(i,K) = i0;
            end
            margin = test_margin;
            if verbose 
                fprintf('    separable using %d samples per cluster: training margin=%1.1e generalization=%1.1e\n', K-1, train_margin, test_margin);
            end
        else
            margin = train_margin;
            if verbose 
                fprintf('    inseparable using %d samples per cluster: training margin=%1.1e\n', K, train_margin);
            end
        end
    end
    separable = margin > 0;
end
% M = 100;
% y=(randi(2, [1, M])-1)*2-1;
% cur = 1;
% for j=0.5:0.25:2.25
%   subplot(4,2,cur); cur=cur+1;
%   x=randn([2, M])+1.8*repmat(y, [2, 1]);
%   scatter(x(1,:), x(2,:), 20, y)
%   [seperable, flag, margin]=check_linear_seperability(x, y);
%   if seperable title('Seperable '); else title('Inseperable'); end
% end
