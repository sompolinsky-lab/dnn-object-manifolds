function R=create_affine_transformation_range(transform_type, scale_factor, n_samples)
    % Create a range of transformation values such that they span the entire range between the 
    % maximal change defined by the scale factor.
    R = [];
    switch transform_type
        case 0
            error('Scale factor is not defined for the identity transform');
        case 1
            r=abs(scale_factor);
        case 2
            r=abs(scale_factor);
        case 3
            R=logspace(log10(1/(1+scale_factor)),log10(1+scale_factor),n_samples);
        case 4
            R=logspace(log10(1/(1+scale_factor)),log10(1+scale_factor),n_samples);
        case 5
            r=abs(scale_factor);
        case 6
            r=abs(scale_factor);
        case 7
            assert(scale_factor/2/sqrt(2)<=1);
            r=2*asin(scale_factor/2/sqrt(2));
        otherwise
            error('Unknown affine transform type: %d', transform_type);
    end
    if isempty(R)
        R = linspace(-r,r,n_samples);
    end
end