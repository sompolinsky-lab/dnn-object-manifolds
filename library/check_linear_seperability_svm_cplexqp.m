%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check for an input data-set of vectors X and labeling y if they are
% linearly seperable by a plane (with bias). 
% 
% This method finds the max-margin linear seperation either in using primal
% or dual formalisms.
%
% Author: Uri Cohen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [separable, best_w, margin, flag, nsv, sv_indices, lagrange_multipliers]=check_linear_seperability_svm_cplexqp(X, y, tolerance, solve_dual, max_iterations)
    if nargin < 3
        tolerance = 1e-10;
    end
    if nargin < 4
        solve_dual = false;
    end
    if nargin < 5
        max_iterations = 0;
    end
    
    M = size(X, 2);
    N = size(X, 1);
    assert(all(size(X) == [N, M]), 'x must be NxM');
    assert(all(size(y) == [1, M]), 'y must be M labels');
    assert(all(abs(y) == 1), 'y must be +1/-1');

    % Remove empty samples
    J = all(isfinite(X),1);
    X = X(:,J);
    y = y(J);
    M = sum(J);
    
    options = cplexoptimset();
    options.Display = 'off';
    %options.qpmethod = 2; 
    name = hostname();
    % Behave well when running in ELSC cluster
    if isempty(name) || contains(name, 'brain') || contains(name, 'ielsc')
        options.threads = 1;
    end

    if max_iterations > 0
        options.simplex.limits.iterations = max_iterations;
    end
    %  The feasibility tolerance specifies the amount by which a solution can violate its 
    % bounds and still be considered feasible. The optimality tolerance specifies the amount 
    % by which a reduced cost in a minimization problem can be negative.
    options.simplex.tolerances.feasibility = tolerance;
    options.simplex.tolerances.optimality = tolerance;

    % Default values for failure
    best_w=nan(N+1,1);
    separable=false;
    margin=nan;
    nsv=0;
    sv_indices=[];
    lagrange_multipliers=[];

    % cplexqp solve quadratic programming problems.
    % x = cplexqp(H,f,Aineq,bineq,Aeq,beq,lb,ub) 
    % solves the quadratic programming problem min 1/2*x'*H*x + f*x 
    % subject to Aineq*x <= bineq with equality constraints Aeq*x = beq
    % and with a set of lower and upper bounds on the design variables, 
    % x, so that the solution is in the range lb <= x <= ub. 
    if solve_dual
        % Specifies the type of solution CPLEX attempts to compute when CPLEX solves 
        % a nonconvex, continuous quadratic model.
        % 2: Search for a solution that satisfies first-order optimality conditions, 
        % but is not necessarily globally optimal.
        options.solutiontarget=2;
        
        Xy = X .* repmat(y, [N, 1]);
        H = Xy'*Xy;
        f = -ones(1, M);
        lb = zeros(1, M);
        Aeq = y; beq = 0;
        [a, L, flag] = cplexqp(H,f,[],[],Aeq,beq,lb,[],[],options); 
        L = -L;
        if strcmp(flag, 'Unknown status') 
            flag = 0;
        end
        if flag >= 0
            assert_warn(L >= 0, sprintf('L=%1.1f flag=%d', L, flag));
            assert(all(a >= 0), 'Negative kkt coefficients found');
            assert(abs(y*a) < 1e-1, sprintf('Bias condition does not hold: %1.1e (flag=%d)', abs(y*a), flag));
            sv_indices = find(a > max(a) * 1e-3);
            lagrange_multipliers = a;
            nsv = length(sv_indices);
            w = Xy * a;

            Xw = X'*w;
            b = mean(Xw(sv_indices) - y(sv_indices)');
            best_w = [w; -b];
            
            Xb = [X; ones(1, M)];
            separable = all(sign(best_w'*Xb)==y);
            
            assert_warn(separable, sprintf('The solution of the dual problem is not seperable (flag=%d)', flag));
        end
    else
        f = zeros(1, N+1);
        H = eye(N+1); H(N+1,N+1) = 0;
        Xy = [X; ones(1, M)] .* repmat(y, [N+1, 1]);
        Aineq = -Xy';
        bineq = -ones(1, M);

        try
            [w, L, flag, output] = cplexqp(H,f,Aineq,bineq,[],[],[],[],[],options);
        catch error
            flag = -100;
            if strfind(error.message, '1256: Basis singular.')
                fprintf('Warning: Basis singular\n');
            else
                warning('cplex failed: %s', error.message);
            end
            return;
        end        

        if ~isempty(w) && all(isfinite(w))
            assert_warn(flag <= 0 || L >= 0, sprintf('Got a negative result L=%1.1f (flag=%d)', L, flag));
            Xw = Xy'*w;
            assert_warn(flag < 0 || flag == 5 || all(Xw - 1 >= -1e-4), sprintf('Violation of kkt conditions: %1.1e (flag=%d, |w|=%1.1e)', min(Xw - 1), flag, norm(w)));
            sv_indices = find((Xw - 1) < 1e-3);
            nsv = length(sv_indices);
            best_w = w;
        end
    end
    
    if isempty(best_w) || ~all(isfinite(best_w))
        flag = -1;
        return;
    end
    assert(all(size(best_w) == [N+1, 1]));
    assert(all(isfinite(best_w)));
    
    Xb = [X; ones(1, M)];
    assert(~all(sign(best_w'*Xb)==-y), 'Sign problem, reversed w');

    separable = all(sign(best_w'*Xb)==y);
    wnorm = norm(best_w(1:N), 2);
    if wnorm == 0
        margin = nan;
    else
        margin = min(best_w'*Xb.*y) / wnorm;
    end
    assert_warn(separable || isnan(margin) || margin < 0, sprintf('Inseparable with positive margin=%1.1e flag=%d(numeric issue)', margin, flag));
    assert(~separable || isnan(margin) || margin >= 0);
end
% M = 100;
% y=(randi(2, [1, M])-1)*2-1;
% cur = 1;
% for j=0.5:0.25:2.25
%   subplot(4,2,cur); cur=cur+1;
%   x=randn([2, M])+1.8*repmat(y, [2, 1]);
%   scatter(x(1,:), x(2,:), 20, y)
%   [separable, w, margin]=check_linear_seperability_svm_cplexqp(x, y);
%   if seperable title('Seperable '); else title('Inseperable'); end
% end
