function generate_convnet_random_change(N_OBJECTS, range_factor, N_SAMPLES, network_type, degrees_of_freedom, N_BATCHES, run_id, epoch, seed, objects_seed)
% Generate manifolds by randomly sampling from multiple directions
if nargin < 6
    N_BATCHES = 4; 
end
if nargin < 7
    run_id = 0;
end
if nargin < 8
    epoch = nan;
end
if nargin < 9
    seed = 0;
end
if nargin < 10
    objects_seed = 0;
end
delete_after_save = false;
validate_after_save = false;

N_COORDINATES = 6;
suffix = '';
if ~isnan(epoch)
    suffix = sprintf('%s_epoch%d', suffix, epoch);
end
if seed ~= 0
    suffix = sprintf('%s_seed%d', suffix, seed);
end
data_suffix = suffix;
if objects_seed ~= 0
    suffix = sprintf('%s_objectsSeed%d', suffix, objects_seed);
end

batch_size = N_OBJECTS / N_BATCHES;
assert(mod(N_OBJECTS, batch_size) == 0);
assert(N_BATCHES == N_OBJECTS/batch_size);

network_type_string = {'alexnet', 'googlenet', 'resnet50', 'resnet18', 'vgg16'};
network_name = network_type_string{network_type};
network_metadata_name = sprintf('convnet_%s_model%s.mat', network_name, data_suffix);
fprintf('Reading network metadata from %s\n', network_metadata_name);
network_metadata = matfile(network_metadata_name);

assert(isprop(network_metadata, 'layer_indices'), 'Indices are not available in network metadata file');
layer_names = network_metadata.layer_names;
layer_sizes = network_metadata.layer_sizes;
layer_indices = network_metadata.layer_indices;
N_LAYERS = network_metadata.N_LAYERS;

prefix = sprintf('%s/generate_%s_random_change_dof%d', network_name, network_name, degrees_of_freedom);
global IMAGENET_IMAGE_SIZE;
if IMAGENET_IMAGE_SIZE ~= 64
    prefix = sprintf('%s_%dpx', prefix, IMAGENET_IMAGE_SIZE);
end
fprintf('Results saved under %s*\n', prefix);
N_TRAIN_OBJECTS = read_imagenet_training_size();

% Object images
global IMAGENET_FRAME_SIZE;
T=tic;
image_indices = choose_imagenet_template_images(N_OBJECTS, objects_seed);
assert(length(image_indices) == N_OBJECTS);
assert(min(image_indices) >= 1 && max(image_indices) <= N_TRAIN_OBJECTS);
img_base = zeros(N_OBJECTS, IMAGENET_FRAME_SIZE, IMAGENET_FRAME_SIZE, 3, 'uint8');
for i=1:N_OBJECTS
    img_base(i,:,:,:) = read_imagenet_thumbnails(image_indices(i));
end
fprintf('Created base images (took %1.1f sec)\n', toc(T));

%img0=single(read_imagenet_thumbnails(img_value))/255;
%T0=create_affine_transform_type(0);

T=tic;
net = convnet_init(network_type, epoch);
if ~isnan(epoch)
    fprintf('Loaded model %d epoch %d (%1.3f sec)\n', network_type, epoch, toc(T));
else
    fprintf('Loaded model %d (%1.3f sec)\n', network_type, toc(T));
end

% Calculate HMAX features of the images
global N_HMAX_FEATURES;
N_NEURONS = N_HMAX_FEATURES;

if degrees_of_freedom == 2
    direction_names = {'translation', 'shear'};
elseif degrees_of_freedom == 4
    direction_names = {'translation and shear'};
else
    assert(degrees_of_freedom == 6);
    direction_names = {'all'};
end
N_DIRECTIONS = length(direction_names);

% Merge files
if length(run_id) > 1
    run_ids = run_id;
    if validate_after_save
        for l=1:N_LAYERS+1
            for run_id = run_ids
                if range_factor < 0.1
                    run_name = sprintf('%s_range%f_P%d_M%d_%s%s_%d.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, layer_names{l}, suffix, run_id);
                else
                    run_name = sprintf('%s_range%1.1f_P%d_M%d_%s%s_%d.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, layer_names{l}, suffix, run_id);
                end
                assert(exist(run_name, 'file')>0, 'File not exists: %s', run_name);
                a = matfile(run_name);
                tf = a.tuning_function;
                if (min(tf(:)) < -1e5)
                    fprintf('Deleting results (min issue) from %s\n', run_name);
                    delete(run_name);
                elseif (max(tf(:)) > 1e5)
                    fprintf('Deleting results (max issue) from %s\n', run_name);
                    delete(run_name);
                end
            end
        end
        return;
    end
    if delete_after_save
        for l=1:N_LAYERS+1
            for run_id = run_ids
                if range_factor < 0.1
                    run_name = sprintf('%s_range%f_P%d_M%d_%s%s_%d.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, layer_names{l}, suffix, run_id);
                else
                    run_name = sprintf('%s_range%1.1f_P%d_M%d_%s%s_%d.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, layer_names{l}, suffix, run_id);
                end
                fprintf('Deleting results loaded from %s\n', run_name);
                delete(run_name);
            end
        end
        return;
    end

    T0=tic;
    for l=1:N_LAYERS+1
        tuning_function = zeros(N_DIRECTIONS, N_OBJECTS, N_SAMPLES, N_NEURONS, 'single');
        sample_coordinates = zeros(N_DIRECTIONS, N_OBJECTS, N_SAMPLES, N_COORDINATES, 'single');
        if range_factor < 0.1
            out_name = sprintf('%s_range%f_P%d_M%d_%s%s.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, layer_names{l}, suffix);
        else
            out_name = sprintf('%s_range%1.1f_P%d_M%d_%s%s.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, layer_names{l}, suffix);
        end
        if exist(out_name, 'file')>0
	    fprintf('Skiping existing: %s\n', out_name);
	    continue;
            %out_file = matfile(out_name);
            %fprintf('Results loaded and will be saved to %s\n', out_name);
	    %n_directions = size(out_file, 'tuning_function',1);
	    %tuning_function(1:n_directions,:,:,:) = out_file.tuning_function;
    	else
            fprintf('Results will be saved to %s\n', out_name);
        end

        tic;
        for run_id = run_ids
            if range_factor < 0.1
                run_name = sprintf('%s_range%f_P%d_M%d_%s%s_%d.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, layer_names{l}, suffix, run_id);
            else
                run_name = sprintf('%s_range%1.1f_P%d_M%d_%s%s_%d.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, layer_names{l}, suffix, run_id);
            end
            fprintf('Reading results from %s\n', run_name);
            assert(exist(run_name, 'file')>0, run_name);
            in_file = matfile(run_name);
            [batch_number, param_id] = ind2sub([N_BATCHES, N_DIRECTIONS], run_id);
            tuning_function(param_id,((batch_number-1)*batch_size+1):(batch_number*batch_size),:,:) = in_file.tuning_function;
            sample_coordinates(param_id,((batch_number-1)*batch_size+1):(batch_number*batch_size),:,:) = in_file.sample_coordinates;
        end
        save(out_name, 'tuning_function', 'image_indices', 'sample_coordinates', 'direction_names', '-v7.3');
        t = toc(T0)*(single(N_LAYERS+1)/l-1);
        if t>1800
            fprintf('Layer %d (took %1.1f seconds) ETA: %1.1f hours\n', l, toc, t / 3600);
        elseif t>30
            fprintf('Layer %d (took %1.1f seconds) ETA: %1.1f minutes\n', l, toc, t / 60);
        else
            fprintf('Layer %d (took %1.1f seconds) ETA: %1.1f seconds\n', l, toc, t);
        end
    end
    return;
end

if run_id == 0
    RUN_DIRECTIONS = 1:N_DIRECTIONS;
    RUN_BATCHES = 1:N_BATCHES;
else
    [batch_number, param_id] = ind2sub([N_BATCHES, N_DIRECTIONS], run_id);
    RUN_DIRECTIONS = param_id;
    RUN_BATCHES = batch_number;
end

rng(seed);
current = 0;
for param_id=RUN_DIRECTIONS
    for batch_number=RUN_BATCHES
        fprintf('Working on %s batch #%d\n', direction_names{param_id}, batch_number);
        T=tic;
        % Variables for current transforms
        pixels_tuning_function = zeros(1, batch_size, N_SAMPLES, N_NEURONS, 'single');
        sample_coordinates = zeros(1, batch_size, N_SAMPLES, N_COORDINATES, 'single');
        convnet_tuning_function = zeros(N_LAYERS, batch_size, N_SAMPLES, N_NEURONS, 'single');

        for i=((batch_number-1)*batch_size+1):(batch_number*batch_size)
            current = current + 1;
            ii = mod(i-1, batch_size)+1;

            tic;
            % Generate samples
            I = zeros([net.meta.normalization.imageSize(1:2), 3, N_SAMPLES], 'single');
            for j=1:N_SAMPLES
                [transform, p] = create_valid_random_affine_transfrom(range_factor, degrees_of_freedom, param_id);
                I0 = calc_imagenet_warp_legacy(squeeze(single(img_base(i,:,:,:))), transform); % Image in the 0..255 range
                sample_coordinates(1,ii,j,:) = p;
                
                % Save grayscale pixel features
                px_data = mean(I0/255,3);
                pixels_tuning_function(1,ii,j,:) = px_data(:);

                % Prepare batch for convnet
                if numel(net.meta.normalization.averageImage) == 3 && isfield(net.meta.normalization, 'imageStd')
                    % PyTorch normalization
                    I1 = imresize(I0, net.meta.normalization.imageSize(1:2));
                    I1 = I1 / 255 ; % scale to (almost) [0,1]
                    I1 = bsxfun(@minus, I1, reshape(net.meta.normalization.averageImage, [1 1 3]));
                    I1 = bsxfun(@rdivide, I1, reshape(net.meta.normalization.imageStd, [1 1 3]));
                else
                    % MatConvNet normalization
                    I1 = imresize(I0, net.meta.normalization.imageSize(1:2))-net.meta.normalization.averageImage;
                end
                I(:,:,:,j) = I1;
            end

            % Feed the samples in batch
            net.mode = 'test';
            net.eval({net.vars(1).name, I});
            % Extract features
            for l=1:N_LAYERS
                s = prod(layer_sizes{l+1});
                actualSize = size(net.vars(l+1).value); assert(actualSize(end)==N_SAMPLES);
                data = reshape(net.vars(l+1).value, [s, N_SAMPLES])';
                I = layer_indices{l};
                convnet_tuning_function(l,ii,:,1:length(I)) = data(:,I);
            end
            f = batch_size / ii - 1;
            t = f*toc(T);
            if t>1800
                fprintf('Direction %d object %d (took %1.1f seconds) ETA: %1.1f hours\n', param_id, i, toc, t / 3600);
            elseif t>30
                fprintf('Direction %d object %d (took %1.1f seconds) ETA: %1.1f minutes\n', param_id, i, toc, t / 60);
            else
                fprintf('Direction %d object %d (took %1.1f seconds) ETA: %1.1f seconds\n', param_id, i, toc, t);
            end
        end
        assert(ii == batch_size);

        T0=tic;
        for l=1:N_LAYERS+1
            tic;
            if range_factor < 0.1
                out_name = sprintf('%s_range%f_P%d_M%d_%s%s_%d.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, layer_names{l}, suffix, run_id);
            else
                out_name = sprintf('%s_range%1.1f_P%d_M%d_%s%s_%d.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, layer_names{l}, suffix, run_id);
            end
            if exist(out_name, 'file')>0
                fprintf('Skipping existing file: %s\n', out_name);
                continue;
            end
            fprintf('Saving %s\n', out_name);

            if l == 1
                tuning_function = pixels_tuning_function;
            else
                tuning_function = convnet_tuning_function(l-1,:,:,:);
            end
            assert(all(isfinite(tuning_function(:))), 'Data with non finite values (layer %d, direction %d)', l, param_id);
            assert(all(abs(tuning_function(:))<1e5), 'Data with large values (layer %d, direction %d): %1.3e', l, param_id, max(tuning_function(:)));
            save(out_name, 'tuning_function', 'image_indices', 'sample_coordinates', 'direction_names', '-v7.3');
            t = toc(T0)*(single(N_LAYERS+1)/l-1);
            if t>1800
                fprintf('Direction %d layer %d (took %1.1f seconds) ETA: %1.1f hours\n', param_id, l, toc, t / 3600);
            elseif t>30
                fprintf('Direction %d layer %d (took %1.1f seconds) ETA: %1.1f minutes\n', param_id, l, toc, t / 60);
            else
                fprintf('Direction %d layer %d (took %1.1f seconds) ETA: %1.1f seconds\n', param_id, l, toc, t);
            end
        end
        fprintf('Results for %s saved (took %1.1f seconds)\n', direction_names{param_id}, toc(T0));
    end
end
end
