function [cost, gradient] = square_corrcoeff_full_cost(V, X)
    [P, N] = size(X);
    K = size(V,2);
    assert(all(size(V) == [N, K])); 
    %assert(norm(V'*V-eye(K))  < 1e-10);
    % Calculate cost
    C = X*X';                       % [P x P]
    c = X*V;                        % [P x K]
    c0 = diag(C) - sum(c.^2, 2);    % [P x 1]
    Fmn = (C-c*c').^2./(c0*c0');    % [P x P]
    cost = sum(Fmn(:))/2;
    
    % Calculate gradient if needed
    if nargout > 1
        X1 = reshape(X, [1, P, N]);
        X2 = reshape(X, [P, 1, N]);
        C1 = reshape(c, [P, 1, 1, K]);
        C2 = reshape(c, [1, P, 1, K]);
        Gmni =      - bsxfun(@times, (C-c*c')./(c0*c0'), bsxfun(@times, C1, X1));
        Gmni = Gmni - bsxfun(@times, (C-c*c')./(c0*c0'), bsxfun(@times, C2, X2));
        Gmni = Gmni + bsxfun(@times, (C-c*c').^2./(c0*c0').^2, bsxfun(@times, c0, bsxfun(@times, C2, X1)));
        Gmni = Gmni + bsxfun(@times, (C-c*c').^2./(c0*c0').^2, bsxfun(@times, c0', bsxfun(@times, C1, X2)));
        gradient = reshape(sum(sum(Gmni, 1), 2), [N, K]);
    end
end