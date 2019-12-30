function image_indices = sample_indices_one_per_category(N_TRAIN_OBJECTS, N_OBJECTS)
    labels_map=read_imagenet_labels();
    used_labels_map=containers.Map;
    
    perm = randperm(N_TRAIN_OBJECTS);
    image_indices = zeros(N_OBJECTS, 1);
    current = 0;
    skipped = 0;
    for i=1:N_TRAIN_OBJECTS
        % Ignore existing categories
        cat = labels_map(perm(i));
        assert(length(cat) == 1);
        cat = cat{1};
        if used_labels_map.isKey(cat)
            %fprintf('%d %d: skipped already used category: %s\n', i, perm(i), cat);
            skipped = skipped+1;
            continue;
        end
        %fprintf('%d %d: found new category: %s\n', i, perm(i), cat);
        % Take the current value
        used_labels_map(cat) = 1;
        current = current + 1;
        image_indices(current) = perm(i);
        if current == N_OBJECTS
            break;
        end
    end
    fprintf('Skipped %d images to choose %d images with unique categories\n', skipped, current);
end