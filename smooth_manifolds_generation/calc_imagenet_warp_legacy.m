function img=calc_imagenet_warp_legacy(img0, T)
    global USE_NEW_IMAGENET_WARP
    if ~isempty(USE_NEW_IMAGENET_WARP)
        img = calc_imagenet_warp(img0, T);
        return;
    end
    % Legacy code to reproduce the croping and sizing of the ImageNet data
    % This is deprecated, use instead calc_imagenet_warp
    global IMAGENET_FRAME_LIMITS;
    global IMAGENET_IMAGE_SIZE;
    IMAGE_DIM = IMAGENET_IMAGE_SIZE; xyWorldLimits = IMAGENET_FRAME_LIMITS;
    reference_frame = imref2d([IMAGE_DIM, IMAGE_DIM], xyWorldLimits, xyWorldLimits);
    img = imwarp(img0, T, 'OutputView', reference_frame);
end
