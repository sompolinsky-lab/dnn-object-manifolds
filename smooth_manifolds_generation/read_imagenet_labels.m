function l = read_imagenet_labels()
    global IMAGENET_IMAGE_SIZE;
    assert(~isempty(IMAGENET_IMAGE_SIZE), 'Run init_imagenet first');
    thumbnails_file = sprintf('imagenet_all_thumbnails_%dpx.mat', IMAGENET_IMAGE_SIZE);
    t = matfile(thumbnails_file);
    l = t.labels;
end