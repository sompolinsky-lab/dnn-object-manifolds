function samples=sample_indices(N, K, R) 
    if nargin < 3
        R = 1;
    end
    % Sample R samples of K of N values
    samples = zeros(R, K);
    for r=1:R
        samples(r,:) = randperm(N, K);
    end
end