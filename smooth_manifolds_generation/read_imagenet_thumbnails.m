function f=read_imagenet_thumbnails(image_id)
    global IMAGENET_IMAGE_SIZE;
    thumbnails_file = sprintf('imagenet_all_thumbnails_%dpx.mat', IMAGENET_IMAGE_SIZE);

    global imagenet_thumbnails;
    N_TRAIN_OBJECTS = read_imagenet_training_size();
    if isempty(imagenet_thumbnails)
        fprintf('Loading ImageNet thumbnails\n');
        t = matfile(thumbnails_file);
        imagenet_thumbnails = t.thumbnails;
        assert(size(imagenet_thumbnails, 4) == N_TRAIN_OBJECTS);
    end
    assert(image_id >= 1 && image_id <= N_TRAIN_OBJECTS, 'Index out of range');
    f=imagenet_thumbnails(:,:,:,image_id);
end