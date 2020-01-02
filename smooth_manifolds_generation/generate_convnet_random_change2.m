function generate_convnet_random_change2(N_OBJECTS, range_factor, N_SAMPLES, network_type, degrees_of_freedom, N_BATCHES, run_id, epoch, n_samples_per_batch, seed, objects_seed)
% Generate manifolds by randomly sampling from multiple directions
% Unlike generate_convnet_random_change, it:
% (1) saves only non-disabled layers,
% (2) supports splitting the samples between runs.
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
    n_samples_per_batch = 801;
end
if nargin < 10
    seed = 0;
end
if nargin < 11
    objects_seed = 0;
end
delete_after_save = true;

N_COORDINATES = 6;
suffix = '';
if ~isnan(epoch)
    suffix = sprintf('%s_epoch%d', suffix, epoch);
end
if seed ~= 0
    suffix = sprintf('%s_seed%d', suffix, seed);
end
if objects_seed ~= 0
    suffix = sprintf('%s_objectsSeed%d', suffix, objects_seed);
end

batch_size = N_OBJECTS / N_BATCHES;
N_BATCH_SAMPLES = ceil(N_SAMPLES / n_samples_per_batch);
assert(mod(N_OBJECTS, batch_size) == 0);
assert(N_BATCHES == N_OBJECTS/batch_size);

layers_grouping_level = 0;
if network_type >= 2 && network_type <= 4
    layers_grouping_level = 2;
end
    
[network_name, N_LAYERS, ~, layer_names, layer_sizes, ...
    ~, ~, ~, ~, ~, ~, layers, ~, ~, layer_indices] = ...
    load_network_metadata(network_type, layers_grouping_level, epoch, seed);
ENABLED_LAYERS = zeros(1, N_LAYERS); ENABLED_LAYERS(layers) = 1;
n_layers = length(layers);
assert(ENABLED_LAYERS(1) == 1, 'The input layer is assumed to be enabled');

%network_type_string = {'alexnet', 'googlenet', 'resnet50', 'resnet18', 'vgg16'};
%network_name = network_type_string{network_type};
%network_metadata_name = sprintf('convnet_%s_model%s.mat', network_name, suffix);
%network_metadata = matfile(network_metadata_name);

%assert(isprop(network_metadata, 'layer_indices'), 'Indices are not available in network metadata file');
%layer_names = network_metadata.layer_names;
%layer_sizes = network_metadata.layer_sizes;
%layer_indices = network_metadata.layer_indices;
%N_LAYERS = network_metadata.N_LAYERS;

if ~exist(network_name, 'dir')
    mkdir(network_name);
end
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

    T0=tic;
    for l=1:N_LAYERS
        if ~ENABLED_LAYERS(l)
            fprintf('Skipping disabled layer %s\n', layer_names{l});
            continue;
        end
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
            [batch_number, param_id, samples_number] = ind2sub([N_BATCHES, N_DIRECTIONS, N_BATCH_SAMPLES], run_id);
            r1 = ((batch_number-1)*batch_size+1):(batch_number*batch_size);
    	    if samples_number*n_samples_per_batch > N_SAMPLES
                dropped_samples = samples_number*n_samples_per_batch - N_SAMPLES;
            else
                dropped_samples = 0;
            end
            r2 = ((samples_number-1)*n_samples_per_batch+1):(samples_number*n_samples_per_batch-dropped_samples);
            R2 = 1:(n_samples_per_batch-dropped_samples);
            a = in_file.tuning_function(:,:,R2,:);
            assert(all(isfinite(a(:))), 'Data with non finite values (id %s, layer %d, direction %d)', run_id, l, param_id);
            tuning_function(param_id,r1,r2,:) = a;
            sample_coordinates(param_id,r1,r2,:) = in_file.sample_coordinates(:,:,R2,:);
        end
        assert(all(isfinite(tuning_function(:))), 'Data with non finite values (layer %d, direction %d)', l, param_id);
        save(out_name, 'tuning_function', 'image_indices', 'sample_coordinates', 'direction_names', '-v7.3');

        if delete_after_save
            if ~ENABLED_LAYERS(l)
                fprintf('Skipping disabled layer %s\n', layer_names{l});
                continue;
            end
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

        t = toc(T0)*(single(N_LAYERS)/l-1);
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

fprintf('Run id range is [1..%d]\n', N_BATCHES*N_DIRECTIONS*N_BATCH_SAMPLES);
if run_id == 0
    RUN_DIRECTIONS = 1:N_DIRECTIONS;
    RUN_BATCHES = 1:N_BATCHES;
else
    [batch_number, param_id, samples_number] = ind2sub([N_BATCHES, N_DIRECTIONS, N_BATCH_SAMPLES], run_id);
    RUN_DIRECTIONS = param_id;
    RUN_BATCHES = batch_number;
    rng(run_id + seed*2^16);
end

current = 0;
for param_id=RUN_DIRECTIONS
    for batch_number=RUN_BATCHES
        fprintf('Working on %s batch #%d (%d layers)\n', direction_names{param_id}, batch_number, n_layers);
        T=tic;
        % Variables for current transforms
        sample_coordinates = zeros(1, batch_size, n_samples_per_batch, N_COORDINATES, 'single');
        convnet_tuning_function = zeros(n_layers, batch_size, n_samples_per_batch, N_NEURONS, 'single');

        for i=((batch_number-1)*batch_size+1):(batch_number*batch_size)
            current = current + 1;
            ii = mod(i-1, batch_size)+1;

            tic;
            % Generate samples
            I = zeros([net.meta.normalization.imageSize(1:2), 3, n_samples_per_batch], 'single');
            for j=1:n_samples_per_batch
                [transform, p] = create_valid_random_affine_transfrom(range_factor, degrees_of_freedom, param_id);
                I0 = calc_imagenet_warp_legacy(squeeze(single(img_base(i,:,:,:))), transform); % Image in the 0..255 range
                sample_coordinates(1,ii,j,:) = p;
                
                % Save grayscale pixel features
                px_data = mean(I0/255,3);
                %assert_warn(all(px_data(:) < 1), sprintf('Found pixel with value %1.1f', max(px_data(:))));
                convnet_tuning_function(1,ii,j,:) = px_data(:);

                % Prepare batch for convnet
                if numel(net.meta.normalization.averageImage) == 3 && isfield(net.meta.normalization, 'imageStd')
                    % PyTorch normalization
                    I1 = imresize(I0, net.meta.normalization.imageSize(1:2));
                    I1 = I1 / 255 ; % scale to (almost) [0,1]
                    I1 = bsxfun(@minus, I1, reshape(net.meta.normalization.averageImage, [1 1 3]));
                    I1 = bsxfun(@rdivide, I1, reshape(net.meta.normalization.imageStd, [1 1 3]));
                else
                    % MatConvNet normalization
                    I1 = imresize(I0, net.meta.normalization.imageSize(1:2))-reshape(net.meta.normalization.averageImage, [1, 1, 3]);
                end
                I(:,:,:,j) = I1;
            end

            % Feed the samples in batch
            net.mode = 'test';
            net.eval({net.vars(1).name, I});
            % Extract features
            for il=2:n_layers
                l = layers(il);
                s = layer_sizes(l);
                actualSize = size(net.vars(l).value); assert(actualSize(end)==n_samples_per_batch);
                data = reshape(net.vars(l).value, [s, n_samples_per_batch])';
                I = layer_indices{l-1};
                convnet_tuning_function(il,ii,:,1:length(I)) = data(:,I);
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
        for il=1:n_layers
            l = layers(il);
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

            tuning_function = convnet_tuning_function(il,:,:,:);
            assert(all(isfinite(tuning_function(:))), 'Data with non finite values (layer %d, direction %d)', l, param_id);
            assert(all(abs(tuning_function(:))<1e5), 'Data with large values (layer %d, direction %d): %1.3e', l, param_id, max(tuning_function(:)));
            save(out_name, 'tuning_function', 'image_indices', 'sample_coordinates', 'direction_names', '-v7.3');
            t = toc(T0)*(single(n_layers)/il-1);
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
