function T=create_random_affine_transform(sigma, degrees_of_freedom, param_id)
    % Sample a random affine transform with the given number of degrees of
    % freedom (supporting 2, 4 and 6);
    A = [1,0,0;0,1,0];
    An = randn(2,3)*sigma+A;
    if degrees_of_freedom == 2
        if param_id == 1
            % Just translation
            An([1, 4]) = A([1, 4]); % no scaling 
            An([2, 3]) = A([2, 3]); % no shear
        else
            assert(param_id == 2);
            % Just shear
            An([1, 4]) = A([1, 4]); % no scaling 
            An([5, 6]) = A([5, 6]); % no translation
        end
    elseif degrees_of_freedom == 4
        % Translation and shear
        An([1, 4]) = A([1, 4]); % no scaling 
        assert(param_id == 1);
    else
        assert(degrees_of_freedom == 6)
        assert(param_id == 1);
    end
    T=affine2d([An;0, 0, 1]');
end