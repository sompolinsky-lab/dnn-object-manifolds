function output_image=calc_imagenet_warp(input_image, transform)
    % Objet size is mapped to the [-1,1] range
    % The input image is then in the [-s,s] for s = frame-size / object-size (surround factor in init_imagenet)
    % The output image is then in the [-o,s] for o = image-size / object-size
    global IMAGENET_IMAGE_SIZE;
    global IMAGENET_FRAME_SIZE;
    global IMAGENET_OBJECT_SIZE;

    input_scale = IMAGENET_FRAME_SIZE / IMAGENET_OBJECT_SIZE;
    xyWorldLimitsInput = [-input_scale, input_scale];
    input_reference_frame = imref2d([IMAGENET_FRAME_SIZE, IMAGENET_FRAME_SIZE], xyWorldLimitsInput, xyWorldLimitsInput);
    output_scale = IMAGENET_IMAGE_SIZE / IMAGENET_OBJECT_SIZE;
    xyWorldLimitsOutout = [-output_scale, output_scale];
    output_reference_frame = imref2d([IMAGENET_IMAGE_SIZE, IMAGENET_IMAGE_SIZE], xyWorldLimitsOutout, xyWorldLimitsOutout);

    % Apply a transformation T to the given image 
    %output_image = imwarp(input_image, input_reference_frame, transform, 'OutputView', output_reference_frame, 'Interp', 'cubic');
    output_image = imwarp(input_image, input_reference_frame, transform, 'OutputView', output_reference_frame);
    assert(size(output_image,1) == IMAGENET_IMAGE_SIZE && size(output_image,2) == IMAGENET_IMAGE_SIZE);
end
