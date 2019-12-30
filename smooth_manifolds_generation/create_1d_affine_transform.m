function transform = create_1d_affine_transform(range_factor, param_id, j, N_SAMPLES)
    global IMAGENET_IMAGE_SIZE;
    global USE_NEW_IMAGENET_WARP; 
    global IMAGENET_OBJECT_SIZE;

    % Affine trasnsformations parameters
    if ~isempty(USE_NEW_IMAGENET_WARP)
        scale_factor = range_factor/(IMAGENET_OBJECT_SIZE/2.);
        translate_x_vec = create_affine_transformation_range(1, scale_factor, N_SAMPLES);
        translate_y_vec = create_affine_transformation_range(2, scale_factor, N_SAMPLES);
        scales_x_vec = create_affine_transformation_range(3, scale_factor, N_SAMPLES);
        scales_y_vec = create_affine_transformation_range(4, scale_factor, N_SAMPLES);
        shear_x_vec = create_affine_transformation_range(5, scale_factor, N_SAMPLES);
        shear_y_vec = create_affine_transformation_range(6, scale_factor, N_SAMPLES);
        angles_vec = create_affine_transformation_range(7, scale_factor, N_SAMPLES);
    elseif IMAGENET_IMAGE_SIZE == 32
        N_TRANSLATE_X = N_SAMPLES; translate_x_vec = linspace(-1., 1., N_TRANSLATE_X)*range_factor;
        N_TRANSLATE_Y = N_SAMPLES; translate_y_vec = linspace(-1., 1., N_TRANSLATE_Y)*range_factor;
        N_SCALES_X = N_SAMPLES; scales_x_vec = exp(linspace(-0.036, 0.036, N_SCALES_X)*range_factor);
        N_SCALES_Y = N_SAMPLES; scales_y_vec = exp(linspace(-0.036, 0.036, N_SCALES_Y)*range_factor);
        N_SHEAR_X = N_SAMPLES; shear_x_vec = linspace(-1, 1, N_SHEAR_X)/28*range_factor;
        N_SHEAR_Y = N_SAMPLES; shear_y_vec = linspace(-1, 1, N_SHEAR_Y)/28*range_factor;
        N_ANGLES = N_SAMPLES; angles_vec = linspace(-3*pi/360,3*pi/360,N_ANGLES)*range_factor;
    elseif IMAGENET_IMAGE_SIZE == 64
        N_TRANSLATE_X = N_SAMPLES; translate_x_vec = linspace(-1., 1., N_TRANSLATE_X)*range_factor;
        N_TRANSLATE_Y = N_SAMPLES; translate_y_vec = linspace(-1., 1., N_TRANSLATE_Y)*range_factor;
        N_SCALES_X = N_SAMPLES; scales_x_vec = exp(linspace(-0.018, 0.018, N_SCALES_X)*range_factor);
        N_SCALES_Y = N_SAMPLES; scales_y_vec = exp(linspace(-0.018, 0.018, N_SCALES_Y)*range_factor);
        N_SHEAR_X = N_SAMPLES; shear_x_vec = linspace(-1, 1, N_SHEAR_X)/52*range_factor;
        N_SHEAR_Y = N_SAMPLES; shear_y_vec = linspace(-1, 1, N_SHEAR_Y)/52*range_factor;
        N_ANGLES = N_SAMPLES; angles_vec = linspace(-1.5*pi/360,1.5*pi/360,N_ANGLES)*range_factor;
    elseif IMAGENET_IMAGE_SIZE == 128
        N_TRANSLATE_X = N_SAMPLES; translate_x_vec = linspace(-1., 1., N_TRANSLATE_X)*range_factor;
        N_TRANSLATE_Y = N_SAMPLES; translate_y_vec = linspace(-1., 1., N_TRANSLATE_Y)*range_factor;
        N_SCALES_X = N_SAMPLES; scales_x_vec = exp(linspace(-0.018, 0.018, N_SCALES_X)*range_factor);
        N_SCALES_Y = N_SAMPLES; scales_y_vec = exp(linspace(-0.018, 0.018, N_SCALES_Y)*range_factor);
        N_SHEAR_X = N_SAMPLES; shear_x_vec = linspace(-1, 1, N_SHEAR_X)/52*range_factor;
        N_SHEAR_Y = N_SAMPLES; shear_y_vec = linspace(-1, 1, N_SHEAR_Y)/52*range_factor;
        N_ANGLES = N_SAMPLES; angles_vec = linspace(-1.5*pi/360,1.5*pi/360,N_ANGLES)*range_factor;
    else
        assert(IMAGENET_IMAGE_SIZE == 224);
        N_TRANSLATE_X = N_SAMPLES; translate_x_vec = linspace(-1., 1., N_TRANSLATE_X)*range_factor;
        N_TRANSLATE_Y = N_SAMPLES; translate_y_vec = linspace(-1., 1., N_TRANSLATE_Y)*range_factor;
        N_SCALES_X = N_SAMPLES; scales_x_vec = exp(linspace(-0.01, 0.01, N_SCALES_X)*0.58*range_factor);    
        N_SCALES_Y = N_SAMPLES; scales_y_vec = exp(linspace(-0.01, 0.01, N_SCALES_Y)*0.52*range_factor);    
        N_SHEAR_X = N_SAMPLES; shear_x_vec = linspace(-1, 1, N_SHEAR_X)/175*range_factor;                   
        N_SHEAR_Y = N_SAMPLES; shear_y_vec = linspace(-1, 1, N_SHEAR_Y)/175*range_factor;                   
        N_ANGLES = N_SAMPLES; angles_vec = linspace(-pi/360,pi/360,N_ANGLES)*0.45*range_factor;             
    end
    
    switch param_id
        case 1
            transform=create_affine_transform_type(param_id, translate_x_vec(j));
        case 2
            transform=create_affine_transform_type(param_id, translate_y_vec(j));
        case 3
            transform=create_affine_transform_type(param_id, scales_x_vec(j));
        case 4
            transform=create_affine_transform_type(param_id, scales_y_vec(j));
        case 5
            transform=create_affine_transform_type(param_id, shear_x_vec(j));
        case 6
            transform=create_affine_transform_type(param_id, shear_y_vec(j));
        case 7
            transform=create_affine_transform_type(param_id, angles_vec(j));
        otherwise
            error('Unknown param-id')
    end
end
