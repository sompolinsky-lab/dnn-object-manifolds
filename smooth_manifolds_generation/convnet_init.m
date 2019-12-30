function net = convnet_init(network_type, epoch)
    if nargin < 2
        epoch = nan;
    end
    % Initialize network from mat files
    
    % Initialize models path
    %path(path, '../MatConvNet');
    run '../MatConvNet/matlab/vl_setupnn'
    modelsDir = '../MatConvNet/models/';
    %modelsDir2 = '../../data/mcnPyTorch/models/';
    if network_type == 1
        if ~isnan(epoch)
            if isinf(epoch)
                modelPath = [modelsDir, 'alexnet-pt-mcn.mat'];
                %modelPath = [modelsDir, 'alexnet-owt-4df8aa71-mcn.mat'];
            else
                modelPath = [modelsDir, sprintf('alexnet_epoch%d-pt-mcn.mat', epoch)];
            end
            % Load network file
            net = dagnn.DagNN.loadobj(load(modelPath));
            
            % Remove Dropout layers
            net.layers(15).outputIndexes = net.layers(16).outputIndexes;
            net.layers(net.layers(15).outputIndexes).inputs = {net.layers(15).name};    
            net.layers(16).outputIndexes=[];
            net.layers(18).outputIndexes = net.layers(19).outputIndexes;
            net.layers(19).outputIndexes=[];
            net.layers(net.layers(18).outputIndexes).inputs = {net.layers(18).name};
            net.removeLayer({'classifier_0', 'classifier_3'});
        else
            modelPath = [modelsDir, 'imagenet-caffe-alex.mat'];
            % Load network file
            net = dagnn.DagNN.fromSimpleNN(load(modelPath));
        end
    elseif network_type == 2
        modelPath = [modelsDir, 'imagenet-googlenet-dag.mat'];
        % Load network file
        net = dagnn.DagNN.loadobj(load(modelPath));
    elseif network_type == 3
        %modelPath = [modelsDir, 'imagenet-resnet-50-dag.mat'];
        modelPath = [modelsDir, 'resnet50-pt-mcn.mat'];
        % Load network file
        net = dagnn.DagNN.loadobj(load(modelPath));
    elseif network_type == 4
        if ~isnan(epoch)
            modelPath = [modelsDir, sprintf('resnet18_epoch%d-pt-mcn.mat', epoch)];
            % Load network file
            net = dagnn.DagNN.loadobj(load(modelPath));
            
            % Remove Dropout layers
            %net.layers(15).outputIndexes = net.layers(16).outputIndexes;
            %net.layers(net.layers(15).outputIndexes).inputs = {net.layers(15).name};    
            %net.layers(16).outputIndexes=[];
            %net.layers(18).outputIndexes = net.layers(19).outputIndexes;
            %net.layers(19).outputIndexes=[];
            %net.layers(net.layers(18).outputIndexes).inputs = {net.layers(18).name};
            %net.removeLayer({'classifier_0', 'classifier_3'});
        else
            modelPath = [modelsDir, 'resnet18-pt-mcn.mat'];
            % Load network file
            net = dagnn.DagNN.loadobj(load(modelPath));
        end
    elseif network_type == 5
        if ~isnan(epoch)
            modelPath = [modelsDir, sprintf('vgg16_epoch%d-pt-mcn.mat', epoch)];
            % Load network file
            net = dagnn.DagNN.loadobj(load(modelPath));
        else
            modelPath = [modelsDir, 'imagenet-vgg-verydeep-16.mat'];
            % Load network file
            net = dagnn.DagNN.fromSimpleNN(load(modelPath));
        end
    elseif network_type == 6
        modelPath = [modelsDir, sprintf('vgg-face.mat')];
        % Load network file
        %net = dagnn.DagNN.loadobj(load(modelPath));
        net = dagnn.DagNN.fromSimpleNN(load(modelPath));
    end

    % Allow extracting values from intermediate layers
    % https://github.com/vlfeat/matconvnet/issues/58
    net.conserveMemory = 0;
end