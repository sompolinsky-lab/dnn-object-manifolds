function dichotomies=create_binary_dichotomies(N_OBJECTS, normalize)
    if nargin < 2
        normalize = false;
    end
    if normalize
        dichotomies = zeros(2^N_OBJECTS, N_OBJECTS);
    else
        dichotomies = zeros(2^N_OBJECTS, N_OBJECTS, 'int8');
    end
    for i=1:2^N_OBJECTS
        classifier_str = dec2bin(i-1,N_OBJECTS);
        classifier_str = classifier_str(N_OBJECTS:-1:1);
        for j=1:N_OBJECTS
            if classifier_str(j) == '1'
                dichotomies(i,j) = 1;
            elseif classifier_str(j) == '0'
                dichotomies(i,j) = -1;
            else
                assert(false);
            end
        end
        if normalize
            I = find(dichotomies(i,:) == 1);
            dichotomies(i,I) = 1./length(I);
            I = find(dichotomies(i,:) == -1);
            dichotomies(i,I) = -1./length(I);
        end
    end
end