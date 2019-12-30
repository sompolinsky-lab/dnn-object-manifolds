%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check for an input data-set of spheres centers and axes and labeling y 
% if they are linearly seperable by a plane (with bias). 
%
% centers is [NxP] and represents centers of P spheres, each of size N. 
% axes is [NxDxP] and represents D axes of from P spheres, each of size N. 
% y is [1xP] and represents the spheres labeling.
%
% This method finds the max-margin linear seperation using either primal
% or dual formalism.
%
% Note this implementation adds the worst item per object, not globally.
% Thus samples_used is samples used per object and max_samples is max
% samples per object (unlike the version with slack variables).
%
% Author: Uri Cohen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [separable, best_w, margin, samples_used, nsv, sv_indices] = ...
    check_linear_seperability_spheres_svm_cplexqp(centers, axes, y, tolerance, ...
        solve_dual, max_iterations, max_samples, verbose)
    % Assert input dimensions
    D = size(axes, 2);    % Samples per cluster
    P = size(axes, 3);    % Number of clusters
    N = size(axes, 1);    % Data size
    assert(all(size(centers) == [N, P]), 'centers must be NxP');
    assert(all(size(axes) == [N, D, P]), 'axes must be NxDxP');
    assert(all(size(y) == [1, P]), 'y must be P labels');
    assert(all(abs(y) == 1), 'y must be +1/-1');
    
    if nargin < 4
        tolerance = 1e-10;
    end
    if nargin < 5
        solve_dual = false;
    end
    if nargin < 6
        max_iterations = 0;
    end
    if nargin < 7
        max_samples = 1000;
    end
    if nargin < 8
        verbose = false;
    end

    % Prepare data with bias
    axes_b = [axes; zeros(1, D, P)];
    assert(all(size(axes_b) == [N+1, D, P]), 'axes must be NxDxP');
    centers_b = [centers; ones(1, P)];
    assert(all(size(centers_b) == [N+1, P]), 'centers must be NxP');
        
    % Look for separability with current indices until data is not
    % separable or problem limits are encountered.
    separable = true;
    test_margin = -1;
    train_margin = 1;
    K = 1;
    currXs = reshape(centers, [N, K, P]);
    while separable && test_margin < 0.99*train_margin && K<=max_samples && test_margin<0
        currX = reshape(currXs, [N, K*P]);
        currY = reshape(repmat(y, [K, 1]), [1, K*P]);
        samples_used = K;
        [separable, best_w, train_margin, ~, nsv, sv_indices] = ...
            check_linear_seperability_svm_cplexqp(currX, currY, tolerance, solve_dual, max_iterations);
        if separable
            wnorm = norm(best_w(1:N), 2);
            assert(isfinite(wnorm));
            test_margin = train_margin;
            % Add the worst sample of each cluster
            %TODO: calculate in matrix form?
            X = nan(N, 1, P);
            for i=1:P
                fvI = all(isfinite(axes_b(:,:,i)),1);
                s = - y(i)*best_w'*axes_b(:,fvI,i);
                if sum(fvI) > 0 && norm(s) > 0
                    s = s / norm(s);
                    X(:,i) = centers(:,i) + axes(:,fvI,i)*s';
                    h0 = y(i)*best_w'*(centers_b(:,i) + axes_b(:,fvI,i)*s')/wnorm;
                    h1 = y(i)*best_w'*(centers_b(:,i) - axes_b(:,fvI,i)*s')/wnorm;
                    assert(h1 - h0 > -1e-10, 'Difference is only %1.3e (%1.3f %1.3f)', h1-h0, h1, h0);
                    test_margin = min(test_margin, h0);
                else X(:,i) = centers(:,i); end
            end
            currXs = cat(2, currXs, X);
            K = K + 1;
            assert(all(size(currXs) == [N, K, P]));
            
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
