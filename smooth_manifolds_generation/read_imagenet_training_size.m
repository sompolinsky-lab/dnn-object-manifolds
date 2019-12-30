function n = read_imagenet_training_size()
    global IMAGENET_IMAGE_SIZE;
    assert(~isempty(IMAGENET_IMAGE_SIZE), 'Run init_imagenet first');
    thumbnails_file = sprintf('imagenet_all_thumbnails_%dpx.mat', IMAGENET_IMAGE_SIZE);
    t = matfile(thumbnails_file);
    ns = size(t, 'thumbnails');
    n = ns(4);
end