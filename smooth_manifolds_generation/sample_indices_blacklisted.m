function image_indices = sample_indices_blacklisted(N_TRAIN_OBJECTS, N_OBJECTS, blacklisted)
    if nargin < 3
        blacklisted = [];
    end
    used_objects=zeros(N_TRAIN_OBJECTS, 1);
    for i=1:length(blacklisted)
        used_objects(blacklisted(i)) = 1;
    end
    perm = randperm(N_TRAIN_OBJECTS);
    image_indices = zeros(N_OBJECTS, 1);
    current = 0;
    skipped = 0;
    for i=1:N_TRAIN_OBJECTS
        % Ignore existing categories
        if used_objects(perm(i))
            %fprintf('%d %d: skipped already used category: %s\n', i, perm(i), cat);
            skipped = skipped+1;
            continue;
        end
        %fprintf('%d %d: found new category: %s\n', i, perm(i), cat);
        % Take the current value
        current = current + 1;
        image_indices(current) = perm(i);
        if current == N_OBJECTS
            break;
        end
    end
    fprintf('Skipped %d images to choose %d images with unique categories\n', skipped, current);
end