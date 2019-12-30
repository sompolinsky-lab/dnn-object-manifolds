function T=create_affine_transform_type(transform_type, value)
    % Create 1d affine transformation of the given type and value
    if nargin < 2
        value = 0;
    end
    switch transform_type
        case 0
            T=affine2d([1, 0, 0;0, 1, 0;0,0,1]);
        case 1
            T=affine2d([1, 0, 0;0, 1, 0;value,0,1]);
        case 2
            T=affine2d([1, 0, 0;0, 1, 0;0,value,1]);
        case 3
            assert(value>0);
            T=affine2d([abs(value), 0, 0;0, 1, 0;0, 0, 1]);
        case 4
            assert(value>0);
            T=affine2d([1, 0, 0;0, abs(value), 0;0, 0, 1]);
        case 5
            T=affine2d([1,0,0;value, 1, 0;0, 0, 1]);
        case 6
            T=affine2d([1,value,0;0, 1, 0;0, 0, 1]);
        case 7
            T=affine2d([cos(value), -sin(value), 0;sin(value), cos(value), 0;0, 0, 1]);
        otherwise
            error('Unknown affine transform type: %d', transform_type);
    end
end