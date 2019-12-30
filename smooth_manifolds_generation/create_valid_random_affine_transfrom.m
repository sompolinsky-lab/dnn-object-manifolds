function [T, p] = create_valid_random_affine_transfrom(range_factor, degrees_of_freedom, param_id)
    global IMAGENET_OBJECT_SIZE;
    global IMAGENET_FRAME_SIZE;
    scale_factor = range_factor/(IMAGENET_OBJECT_SIZE/2.);
    ones_image = ones(IMAGENET_FRAME_SIZE, IMAGENET_FRAME_SIZE);
    img = 0; iter = 0;
    while sum(img(:)<1)>0
        iter = iter+1;
        if mod(iter, 10) == 0
            fprintf('Required %d iterations\n', iter);
        end
        T = create_random_affine_transform_bounded(scale_factor, degrees_of_freedom, param_id);
        img = calc_imagenet_warp(ones_image, T);
    end
    p = T.T(1:6);
end
