function [network_name, N_LAYERS, ACTIVE_LAYERS, layer_names, layer_sizes, type_markers, layer_type_markers, parent_ids, multi_parent_ids, ...
    ACTIVE_MARKERS, active_type_markers_names, layers, layer_types, all_type_marker_names, layer_indices, layer_dimensions] = ...
    load_network_metadata(network_type, layers_grouping_level, epoch, seed)
    if nargin < 3
        epoch = nan;
    end
    if nargin < 4
        seed = 0;
    end

% Load metadata
network_type_string = {'hmax', 'alexnet', 'googlenet', 'resnet50', 'resnet18', 'vgg16', 'vggface'};
network_name = network_type_string{network_type+1};
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
network_metadata_name = sprintf('convnet_%s_model%s.mat', network_name, suffix);
fprintf('Reading network metadata from %s\n', network_metadata_name);

network_metadata = matfile(network_metadata_name);
layer_names = network_metadata.layer_names;
layer_types = network_metadata.layer_types;
layer_parent_ids = network_metadata.layer_parent_ids;
layer_dimensions = network_metadata.layer_sizes;
layer_indices = network_metadata.layer_indices;
N_LAYERS = network_metadata.N_LAYERS+1;

ACTIVE_LAYERS = 1:N_LAYERS;
%if strcmp(layer_types{N_LAYERS}, 'dagnn.SoftMax')
%    ACTIVE_LAYERS = 1:(N_LAYERS-1);
%end

single_parent_ids = nan(1, N_LAYERS);
unit_parent_ids = nan(1, N_LAYERS);
coarse_parent_ids = nan(1, N_LAYERS);
layer_type_id = zeros(1, N_LAYERS);
multi_parent_ids = layer_parent_ids;
leafs = ones(size(layer_parent_ids));
unit_multi_parent_ids = layer_parent_ids;

% If mergind ReLU with previous layers, use the previous layers' markers
type_markers = ['o', '>', 'd', 's', '^', 'V', '<', '*', 'h', 'p', '^', 'p', '<', '^', '^', '<', '^'];
if bitand(layers_grouping_level, 1) == 1
    type_markers = ['o', '>', 'd', 's', 's', 'V', '<', '*', 'h', 'h', '^', 'p', '<', '<', 'V', '<', '<'];
end
%pooling_type_markers = ['>', 'd'];
%linear_type_markers = ['d', 's', 'h'];
%nonlinear_type_markers = ['^', '*', 'p'];
%normalization_type_markers = ['V'];
%compound_type_markers = ['<'];
type_markers_names = {'Input', 'Max Pooling', 'Average Pooling', 'Convolution', 'ReLU (after Conv)', 'LRN', 'Concat', 'SoftMax', 'FC', 'ReLU (after FC)', 'Gabor', 'RBF', 'Sum', 'ReLU (after Sum)', 'ReLU (after LRN)', 'Downsample', 'ReLU (after Downsample)'};
assert(length(type_markers_names) == length(type_markers));

layer_sizes = zeros(1, N_LAYERS);
for l=1:N_LAYERS
    % If there is only a single parent, always use it
    if length(layer_parent_ids{l}) == 1 && isfinite(layer_parent_ids{l})
        single_parent_ids(l) = layer_parent_ids{l};
        unit_parent_ids(l) = layer_parent_ids{l};
    end
    
    % Update the parents are not leafs
    if isfinite(layer_parent_ids{l})
        leafs(layer_parent_ids{l}) = 0;
    end
    
    % Remove from the tree skipped parents
    if sum(isfinite(layer_parent_ids{l}))>0
        for ij=find(layer_type_id(layer_parent_ids{l}) == 0)
            j = layer_parent_ids{l}(ij);
            unit_parent_ids(l) = unit_parent_ids(j);
            single_parent_ids(l) = single_parent_ids(j);
            unit_multi_parent_ids{l} = unit_multi_parent_ids{j};
            multi_parent_ids{l} = multi_parent_ids{j};
            unit_parent_ids(j) = nan;
            single_parent_ids(j) = nan;
            unit_multi_parent_ids{j} = [];
            multi_parent_ids{j} = [];
        end
    end
    
    % HMAX layers: ['o', '^', 'd', 's', 'd', 's', 'd', 's', 'd'];
    s = layer_dimensions{l};
    layer_sizes(l) = prod(s);
    if strcmp(layer_types{l}, 'Input')
        assert(l == 1);
        layer_type_id(l) = 1;
    elseif strncmp(layer_types{l}, 'dagnn.Pooling', length('dagnn.Pooling'))
        if strcmp(layer_types{l}, 'dagnn.Pooling (max)')
            layer_type_id(l) = 2;
        else
            assert(strcmp(layer_types{l}, 'dagnn.Pooling (avg)'), sprintf('Unknown pooling layer type: %s', layer_types{l}));
            layer_type_id(l) = 3;
        end
    elseif strcmp(layer_types{l}, 'dagnn.Conv') && ~all(s(1:2) == [1, 1])
        assert(strcmp(layer_types{l+1}, 'dagnn.ReLU') || strcmp(layer_types{l+1}, 'dagnn.SoftMax') || strcmp(layer_types{l+1}, 'dagnn.BatchNorm'), ...
            'A convolution followed by %s: %s', layer_types{l+1}, layer_names{l});
        %assert(layers_grouping_level == 0 || strcmp(layer_types{l+1}, 'dagnn.ReLU') || strcmp(layer_types{l+1}, 'dagnn.SoftMax'), layer_names{l});
        layer_type_id(l) = 4;
    elseif strcmp(layer_types{l}, 'dagnn.ReLU')
        %assert(strcmp(layer_types{l-1}, 'dagnn.Conv'), layer_names{l});

        % If the first bit in layers_grouping_level is set, group ReLU and previous convolution
        if ((network_type < 3) || (network_type >= 5)) && (bitand(layers_grouping_level, 1) == 1)
            assert(strcmp(layer_types{l-1}, 'dagnn.Conv'), 'Layer %s has previous %s', layer_names{l}, layer_types{l-1});
            if strcmp(layer_types{l-1}, 'dagnn.Conv')
                unit_parent_ids(l) = unit_parent_ids(l-1);
                unit_multi_parent_ids{l} = unit_multi_parent_ids{l-1};
                unit_parent_ids(l-1) = nan;
                unit_multi_parent_ids{l-1} = [];
            end
        end
        % In ResNet, if the first bit in layers_grouping_level is set, group ReLU and previous LRN
        if ((network_type == 3) || (network_type == 4)) && (bitand(layers_grouping_level, 1) == 1)
            assert(strcmp(layer_types{l-1}, 'dagnn.BatchNorm') || strcmp(layer_types{l-1}, 'dagnn.Sum'), 'Layer %s has previous %s', layer_names{l}, layer_types{l-1});
            if strcmp(layer_types{l-1}, 'dagnn.BatchNorm')
                unit_parent_ids(l) = unit_parent_ids(l-1);
                unit_multi_parent_ids{l} = unit_multi_parent_ids{l-1};
                unit_parent_ids(l-1) = nan;
                unit_multi_parent_ids{l-1} = [];
            end
        end
        if layer_type_id(l-1) == 4
            layer_type_id(l) = 5;
        elseif layer_type_id(l-1) == 9
            layer_type_id(l) = 10;
        elseif layer_type_id(l-1) == 13
            layer_type_id(l) = 14;
        elseif layer_type_id(l-1) == 6
            layer_type_id(l) = 15;
        else
            assert(false, 'Unsupported valud before ReLU: %d\n', layer_type_id(l-1));
        end
        %if layer_type_id(l-1) == 3 || layer_type_id(l-1) == 5 || layer_type_id(l-1) == 10
        %    layer_type_id(l) = 4;
        %else
        %    assert(layer_type_id(l-1) == 8, 'ReLU unit after unexpected layer');
        %    layer_type_id(l) = 9;
        %end
    elseif strcmp(layer_types{l}, 'dagnn.LRN') || strcmp(layer_types{l}, 'dagnn.BatchNorm')
        layer_type_id(l) = 6;
    elseif strcmp(layer_types{l}, 'dagnn.Concat')
        layer_type_id(l) = 7;
    elseif strcmp(layer_types{l}, 'dagnn.SoftMax')
        layer_type_id(l) = 8;
    elseif strcmp(layer_types{l}, 'dagnn.Conv') && all(s(1:2) == [1, 1])
        layer_type_id(l) = 9;
    elseif strcmp(layer_types{l}, 'dagnn.Gabor')
        layer_type_id(l) = 11;
    elseif strcmp(layer_types{l}, 'dagnn.RBF')
        layer_type_id(l) = 12;
    elseif strcmp(layer_types{l}, 'dagnn.Sum')
        layer_type_id(l) = 13;
    else
        layer_type_id(l) = 0;
        if ~strcmp(layer_types{l}, 'dagnn.Flatten') && ~strcmp(layer_types{l}, 'dagnn.Permute')
            fprintf('Warning: ignoring layer #%d: %s (%s)\n', l, layer_names{l}, layer_types{l});
        end
    end
    if network_type == 3 || network_type == 4
        % In ResNet all loops are summations
        if layer_type_id(l) == 13 
            assert(length(layer_parent_ids{l}) == 2, 'Layer %d has %d parents', layer_type_id(l), length(layer_parent_ids{l}));
        else
            assert(length(layer_parent_ids{l}) == 1, 'Layer %d has %d parents', layer_type_id(l), length(layer_parent_ids{l}));
        end
    end
end
extended_type_markers = [' ', type_markers];
layer_type_markers = extended_type_markers(layer_type_id+1);

if network_type == 3 || network_type == 4
    sum_layers=find(layer_type_id==13);
    coarse_parent_ids(1:5)=unit_parent_ids(1:5);
    previous_sum_layers = [5, sum_layers+1];
    %coarse_parent_ids(sum_layers(1))=5;
    for ic=1:length(sum_layers)
        l = sum_layers(ic);
        p = layer_parent_ids{l};
        assert(length(p) == 2);
        assert(isnan(single_parent_ids(l)));
        %coarse_parent_ids(sum_layers(ic-1)+1:sum_layers(ic)-1)=unit_parent_ids(sum_layers(ic-1)+1:sum_layers(ic)-1);
        if ic > 1
            coarse_parent_ids(sum_layers(ic-1)+1)=sum_layers(ic-1);
        end
        pp = sum_layers(ic);
        if isempty(find(multi_parent_ids{pp}==previous_sum_layers(ic), 1))
            assert(layer_type_id(pp) == 13);
            layer_type_id(pp) = 16;
            assert(layer_type_id(pp+1) == 14);
            layer_type_id(pp+1) = 17;
        end
        coarse_parent_ids(pp) = previous_sum_layers(ic);
        %while multi_parent_ids{pp}(1)>=previous_sum_layers(ic)
        %    coarse_parent_ids(pp) = multi_parent_ids{pp}(1);
        %    pp = multi_parent_ids{pp}(1);
        %end        
        %single_parent_ids(l) = p(2);
    end
    coarse_parent_ids(sum_layers(ic)+1:N_LAYERS)=unit_parent_ids(sum_layers(ic)+1:N_LAYERS);
end


if network_type == 2
    % In GoogLeNet all loops are concatenations
    concat_layers=find(layer_type_id==7);
    for ic=2:length(concat_layers)
        coarse_parent_ids(concat_layers(ic))=concat_layers(ic-1);
    end
    coarse_parent_ids(concat_layers(1))=11;
    coarse_parent_ids(152:154) = unit_parent_ids(152:154);
    coarse_parent_ids(103:108) = unit_parent_ids(103:108);
    coarse_parent_ids(55:60) = unit_parent_ids(55:60);
    coarse_parent_ids(1:11) = unit_parent_ids(1:11);
end

single_parent_ids = single_parent_ids(ACTIVE_LAYERS);
unit_parent_ids = unit_parent_ids(ACTIVE_LAYERS);
coarse_parent_ids = coarse_parent_ids(ACTIVE_LAYERS);
if network_type == 1 || network_type >= 5
    if layers_grouping_level == 0
        parent_ids = single_parent_ids;
    elseif bitand(layers_grouping_level, 1) == 1
        parent_ids = unit_parent_ids;
        multi_parent_ids = unit_multi_parent_ids;
    end
elseif network_type >= 2 && network_type <= 4
    if layers_grouping_level == 0
        parent_ids = single_parent_ids;
    elseif layers_grouping_level == 1
        parent_ids = unit_parent_ids;
        multi_parent_ids = unit_multi_parent_ids;
    elseif bitand(layers_grouping_level, 2) == 2
        parent_ids = coarse_parent_ids;
        multi_parent_ids = coarse_parent_ids;
    end
end

layer_names = layer_names(ACTIVE_LAYERS);
layer_sizes = layer_sizes(ACTIVE_LAYERS);
layer_type_markers = layer_type_markers(ACTIVE_LAYERS);
%layer_parent_ids = layer_parent_ids(ACTIVE_LAYERS);

all_parents = [];
for l=1:N_LAYERS
    if iscell(multi_parent_ids)
        all_parents = [all_parents, multi_parent_ids{l}];
    else
        all_parents = [all_parents, multi_parent_ids(l)];
    end
end
all_parents = unique(all_parents(isfinite(all_parents)));
ACTIVE_MARKERS = [];
for i=1:length(type_markers)
    %if sum(layer_type_markers(isfinite(parent_ids))==type_markers(i))>0
    if sum(layer_type_id(all_parents)==i)>0
        ACTIVE_MARKERS = [ACTIVE_MARKERS, i];
    end
end
unit_type_marker_names = type_markers_names;
if bitand(layers_grouping_level, 1) == 1
    unit_type_marker_names{5} = 'Conv+ReLU';
    unit_type_marker_names{10} = 'FC+ReLU';
end
if network_type == 2 && bitand(layers_grouping_level, 2) == 2
    unit_type_marker_names{7} = 'Inception';
end
if (network_type == 3 || network_type == 4) && bitand(layers_grouping_level, 2) == 2
    unit_type_marker_names{13} = 'Skip';
    unit_type_marker_names{14} = 'ReLU (after Skip)';
    unit_type_marker_names{16} = 'Downsample';
    unit_type_marker_names{17} = 'ReLU (after Downsample)';
    % Post-processing to leave only ReLU after Skip, not the skip layers themselves
    if bitand(layers_grouping_level, 1) == 1
        unit_type_marker_names{15} = 'LRN+ReLU';

        skipI = find(layer_type_id==13);
        reluI = find(layer_type_id==14);
        assert(length(skipI)==length(reluI) & all(skipI+1==reluI));
        assert(all(parent_ids(reluI) == skipI));
        parent_ids(reluI) = parent_ids(skipI);
        parent_ids(skipI) = nan;
        multi_parent_ids(reluI) = multi_parent_ids(skipI);
        multi_parent_ids(skipI) = nan;
        unit_type_marker_names{14} = 'Skip+ReLU';

        skipI = find(layer_type_id==16);
        reluI = find(layer_type_id==17);
        assert(length(skipI)==length(reluI) & all(skipI+1==reluI));
        assert(all(parent_ids(reluI) == skipI));
        parent_ids(reluI) = parent_ids(skipI);
        parent_ids(skipI) = nan;
        multi_parent_ids(reluI) = multi_parent_ids(skipI);
        multi_parent_ids(skipI) = nan;
        unit_type_marker_names{16} = 'Down+ReLU';
end
end
all_type_marker_names = unit_type_marker_names;

active_type_markers_names = unit_type_marker_names(ACTIVE_MARKERS);
layers = sort(union(parent_ids(isfinite(parent_ids)), find(isfinite(parent_ids))));
layers=layers(layer_type_id(layers)>0);
%layer_types = unit_type_marker_names(layer_type_id);
layer_types = cell(size(layer_type_id));
layer_types(layer_type_id>0) = unit_type_marker_names(layer_type_id(layer_type_id>0));

end
