function generate_convnet_model_metadata(network_type, epoch, seed)
    if nargin < 2
        epoch = nan;
    end
    if nargin < 3
        seed = 0;
    end
    % Generate information needed for convnet processing

    network_type_string = {'alexnet', 'googlenet', 'resnet50', 'resnet18', 'vgg16', 'vggface'};
    network_name = network_type_string{network_type};
    suffix = '';
    global IMAGENET_IMAGE_SIZE;
    if IMAGENET_IMAGE_SIZE ~= 64
        suffix = sprintf('%s_%dpx', suffix, IMAGENET_IMAGE_SIZE);
    end
    if ~isnan(epoch)
        suffix = sprintf('%s_epoch%d', suffix, epoch);
    end
    if seed ~= 0
        suffix = sprintf('%s_seed%d', suffix, seed);
    end
    out_name = sprintf('convnet_%s_model%s.mat', network_name, suffix);
    
    net = convnet_init(network_type, epoch);
    I0 = single(read_imagenet_thumbnails(1)); % Image in the 0..255 range
    if numel(net.meta.normalization.averageImage) == 3 && isfield(net.meta.normalization, 'imageStd')
        % PyTorch normalization
        I = imresize(I0, net.meta.normalization.imageSize(1:2));
        I = I / 255 ; % scale to (almost) [0,1]
        I = bsxfun(@minus, I, reshape(net.meta.normalization.averageImage, [1 1 3]));
        I = bsxfun(@rdivide, I, reshape(net.meta.normalization.imageStd, [1 1 3]));
    else
        % MatConvNet normalization
        I = imresize(I0, net.meta.normalization.imageSize(1:2))-net.meta.normalization.averageImage;
    end
    net.mode = 'test';
    net.eval({net.vars(1).name, I}) ;
    global N_HMAX_FEATURES;
    assert(~isempty(N_HMAX_FEATURES), 'run init_imagenet first');
    
    assert(length(net.vars) == length(net.layers)+1);
    N_LAYERS = length(net.layers);
    layer_names = cell(1,N_LAYERS+1);
    layer_types = cell(1,N_LAYERS+1);
    layer_sizes = cell(1,N_LAYERS+1);
    layer_parent_ids = cell(1,N_LAYERS+1);

    n_features = IMAGENET_IMAGE_SIZE*IMAGENET_IMAGE_SIZE;
    if n_features  < N_HMAX_FEATURES
        pixel_indices = 1:n_features; % use all features
        %layer_projections{i} = eye(n_features);
    else
        pixel_indices = sample_indices(n_features, N_HMAX_FEATURES, 1);
        %layer_projections{i} = randn(n_features, N_HMAX_FEATURES);
    end        

    %layer_projections = cell(1,N_LAYERS);
    layer_indices = cell(1,N_LAYERS);
    
    name_to_id_map = containers.Map;
    layer_names{1} = net.vars(1).name;
    name_to_id_map(layer_names{1}) = 1;
    layer_types{1} = 'Input';
    layer_sizes{1} = size(net.vars(1).value);
    layer_parent_ids{1} = nan;
    
    rng(seed);
    for i=1:N_LAYERS
        %assert(strcmp(net.vars(i+1).name, net.layers(i).name));
        layer_names{i+1} = net.layers(i).name;
        name_to_id_map(net.vars(i+1).name) = i+1;
        layer_types{i+1} = class(net.layers(i).block);
        if strcmp(layer_types{i+1}, 'dagnn.Pooling')
            layer_types{i+1} = sprintf('%s (%s)', class(net.layers(i).block), net.layers(i).block.method);
        end
        layer_sizes{i+1} = size(net.vars(i+1).value);
        for pid=1:length(net.layers(i).inputs)
            parent_name = net.layers(i).inputs{pid};
            assert(name_to_id_map.isKey(parent_name), parent_name);
            if pid == 1
                layer_parent_ids{i+1} = name_to_id_map(parent_name);
            else
                layer_parent_ids{i+1} = [layer_parent_ids{i+1}, name_to_id_map(parent_name)];
            end
        end
        n_features = numel(net.vars(i+1).value);
        fprintf('%s (aka %s, below %s) [%s] %d features\n', net.layers(i).name, net.vars(i+1).name, parent_name, layer_types{i+1}, n_features);
        if n_features  < N_HMAX_FEATURES
            layer_indices{i} = 1:n_features; % use all features
            %layer_projections{i} = eye(n_features);
        else
            layer_indices{i} = sample_indices(n_features, N_HMAX_FEATURES, 1);
            %layer_projections{i} = randn(n_features, N_HMAX_FEATURES);
        end        
    end
    save(out_name, 'N_LAYERS', 'layer_names', 'layer_types', 'layer_sizes', 'layer_indices', 'pixel_indices', 'layer_parent_ids', '-v7.3');
end
