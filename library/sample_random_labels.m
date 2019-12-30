function y=sample_random_labels(N_OBJECTS, random_labeling_type)
    % Sample labels using the given scheme:
    % 0: iid, 1: balanced, 2: sparse
    if random_labeling_type == 2    % Sparse labeling
        y = ones(1,N_OBJECTS); 
        y(randi(N_OBJECTS, 1))=-1;
    elseif random_labeling_type == 1    % Balanced labeling
        y = ones(1,N_OBJECTS); 
        y(randperm(N_OBJECTS, round(N_OBJECTS/2)))=-1;
        assert(abs(sum(y)) <= 1, 'Non balanced labeling');
    else                                % IID labeling
        y = 2*randi([0, 1], [1,N_OBJECTS])-1;
    end
end