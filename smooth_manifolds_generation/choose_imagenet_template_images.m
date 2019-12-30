function image_indices=choose_imagenet_template_images(N_OBJECTS, random_seed)
    base_seed = 0;
    if nargin < 2
        if N_OBJECTS == 128
            random_seed = 1;
            base_seed = 1;
        elseif N_OBJECTS == 32
            random_seed = 4;
        elseif N_OBJECTS == 16
            random_seed = 4;
        else
            random_seed = 0;
        end
    end
    N_TRAIN_OBJECTS = read_imagenet_training_size();
    if random_seed == base_seed
        rng(random_seed);
        image_indices = sample_indices_one_per_category(N_TRAIN_OBJECTS, N_OBJECTS);
        return;
    end
    
    % Sampled object images
    blacklisted = [];
    for previous_seed=base_seed:(random_seed-1)
        rng(previous_seed);
        if previous_seed == base_seed
            previous_indices = sample_indices_one_per_category(N_TRAIN_OBJECTS, N_OBJECTS);
        else
            previous_indices = sample_indices_blacklisted(N_TRAIN_OBJECTS, N_OBJECTS, blacklisted);
        end
        blacklisted = [blacklisted(:); previous_indices(:)];
    end
    rng(random_seed);
    image_indices = sample_indices_blacklisted(N_TRAIN_OBJECTS, N_OBJECTS, blacklisted); 
    % Old code which didn't enforce one-per-category:
    % image_indices = sample_indices(N_TRAIN_OBJECTS, N_OBJECTS, 1);
end