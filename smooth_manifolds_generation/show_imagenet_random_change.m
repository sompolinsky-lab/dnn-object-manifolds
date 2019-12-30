function show_imagenet_random_change(N_OBJECTS, range_factor, degrees_of_freedom, image_index)
% Generate manifolds by randomly sampling from multiple directions
n_rows = 6; 
n_columns = 6;
N_SAMPLES = n_rows * n_columns;
prefix = sprintf('show_imagenet_random_change_dof%d', degrees_of_freedom);
global IMAGENET_IMAGE_SIZE;
if IMAGENET_IMAGE_SIZE ~= 64
    prefix = sprintf('%s_%dpx', prefix, IMAGENET_IMAGE_SIZE);
end
N_TRAIN_OBJECTS = read_imagenet_training_size();

% Object images
global IMAGENET_FRAME_SIZE;
T=tic;
image_indices = choose_imagenet_template_images(N_OBJECTS);
assert(length(image_indices) == N_OBJECTS);
assert(min(image_indices) >= 1 && max(image_indices) <= N_TRAIN_OBJECTS);
img_base = zeros(N_OBJECTS+1, IMAGENET_FRAME_SIZE, IMAGENET_FRAME_SIZE, 3, 'uint8');
img_base(1,:,:,:) = imread('../../docs/results/dog_small.jpg');
for i=1:N_OBJECTS
    img_base(i+1,:,:,:) = read_imagenet_thumbnails(image_indices(i));
end
fprintf('Created base images (took %1.1f sec)\n', toc(T));

%img0=single(read_imagenet_thumbnails(img_value))/255;
%T0=create_affine_transform_type(0);

% Calculate HMAX features of the images
global N_HMAX_FEATURES;
N_NEURONS = N_HMAX_FEATURES;
N_DIM = sqrt(N_NEURONS);

if degrees_of_freedom == 2
    direction_names = {'translation', 'shear'};
elseif degrees_of_freedom == 4
    direction_names = {'translation and shear'};
else
    assert(degrees_of_freedom == 6);
    direction_names = {'all'};
end
N_DIRECTIONS = length(direction_names);
global IMAGENET_OBJECT_SIZE;
scale_factor = range_factor/(IMAGENET_OBJECT_SIZE/2.);
global IMAGENET_FRAME;
global IMAGENET_FRAME_LIMITS;

for param_id=1:N_DIRECTIONS
    fprintf('Working on %s\n', direction_names{param_id});
    out_filename = sprintf('%s_obj%d_range%1.1f_%dx%d_%s', prefix, image_index, range_factor, n_rows, n_columns, direction_names{param_id});
    fprintf('Results saved to %s\n', prefix);

    % Variables for current transforms
    pixels_grayscale_tuning_function = zeros(N_SAMPLES, N_DIM, N_DIM, 'single');
    pixels_rdb_tuning_function = zeros(N_SAMPLES, N_DIM, N_DIM, 3, 'single');

    % Generate samples
    img0 = squeeze(single(img_base(image_index+1,:,:,:)));
    coordinates = zeros(N_SAMPLES, 2);
    for j=1:N_SAMPLES
        % Save grayscale pixel features
        displacement = 0;
        %while displacement < scale_factor*0.9
        while displacement == 0
            [transform, ~] = create_valid_random_affine_transfrom(range_factor, degrees_of_freedom, param_id);
            displacement = calc_affine_transform_magnitude(transform);
        end
        if param_id == 1
            coordinates(j,:) = transform.T([3,6]);
        elseif param_id == 2
            coordinates(j,:) = transform.T([2,4]);
        end
        
        %fprintf('%1.3f\n', displacement);
        I0 = calc_imagenet_warp_legacy(img0, transform); % Image in the 0..255 range
        pixels_rdb_tuning_function(j,:,:,:) = I0/255;
        pixels_grayscale_tuning_function(j,:,:) = mean(I0/255,3);
    end
    
    img = zeros((N_DIM+3)*n_rows+3, (N_DIM+3)*n_columns+3, 3);
    current = 0;
    for i=1:n_rows
        for j=1:n_columns
            current = current + 1;
            img((N_DIM+3)*(i-1)+4:(N_DIM+3)*(i-1)+N_DIM+3, (N_DIM+3)*(j-1)+4:(N_DIM+3)*(j-1)+N_DIM+3,:) = pixels_rdb_tuning_function(current,:,:,:);
        end
    end
    imwrite(img,[out_filename, '.png']);
    
    img0 = squeeze(single(img_base(image_index+1,:,:,:)))/255;
    a=IMAGENET_FRAME_LIMITS(1)+IMAGENET_FRAME-1;
    b=IMAGENET_FRAME_LIMITS(2)-IMAGENET_FRAME+1;
    img0(a,a:b)=1; img0(b,a:b)=1; img0(a:b,a)=1; img0(a:b,b)=1;
    a=IMAGENET_FRAME_LIMITS(1)+IMAGENET_FRAME;
    b=IMAGENET_FRAME_LIMITS(2)-IMAGENET_FRAME;
    img0(a,a:b)=1; img0(b,a:b)=1; img0(a:b,a)=1; img0(a:b,b)=1;
    T0=create_affine_transform_type(0);
    template =  calc_imagenet_warp_legacy(img0, T0);

    LARGEST = 3; SMALLEST = 1;
    Txt1=create_1d_affine_transform(range_factor, 1, SMALLEST, LARGEST);
    Txt2=create_1d_affine_transform(range_factor, 1, LARGEST, LARGEST);
    Tyt1=create_1d_affine_transform(range_factor, 2, SMALLEST, LARGEST);
    Tyt2=create_1d_affine_transform(range_factor, 2, LARGEST, LARGEST);
    %Txs1=create_1d_affine_transform(range_factor, 3, SMALLEST, LARGEST);
    %Txs2=create_1d_affine_transform(range_factor, 3, LARGEST, LARGEST);
    %Tys1=create_1d_affine_transform(range_factor, 4, SMALLEST, LARGEST);
    %Tys2=create_1d_affine_transform(range_factor, 4, LARGEST, LARGEST);
    Txr1=create_1d_affine_transform(range_factor, 5, SMALLEST, LARGEST);
    Txr2=create_1d_affine_transform(range_factor, 5, LARGEST, LARGEST);
    Tyr1=create_1d_affine_transform(range_factor, 6, SMALLEST, LARGEST);
    Tyr2=create_1d_affine_transform(range_factor, 6, LARGEST, LARGEST);
    %Trt1=create_1d_affine_transform(range_factor, 7, SMALLEST, LARGEST);
    %Trt2=create_1d_affine_transform(range_factor, 7, LARGEST, LARGEST);    
    xt1 = calc_imagenet_warp_legacy(img0, Txt1);
    xt2 = calc_imagenet_warp_legacy(img0, Txt2);
    yt1 = calc_imagenet_warp_legacy(img0, Tyt1);
    yt2 = calc_imagenet_warp_legacy(img0, Tyt2);
    %xs1 = calc_imagenet_warp_legacy(img0, Txs1);
    %xs2 = calc_imagenet_warp_legacy(img0, Txs2);
    %ys1 = calc_imagenet_warp_legacy(img0, Tys1);
    %ys2 = calc_imagenet_warp_legacy(img0, Tys2);
    xr1 = calc_imagenet_warp_legacy(img0, Txr1);
    xr2 = calc_imagenet_warp_legacy(img0, Txr2);
    yr1 = calc_imagenet_warp_legacy(img0, Tyr1);
    yr2 = calc_imagenet_warp_legacy(img0, Tyr2);
    %rt1 = calc_imagenet_warp_legacy(img0, Trt1);
    %rt2 = calc_imagenet_warp_legacy(img0, Trt2);

    path(path, [pwd, '/../figures/'])
    [Tableau10, Tableau20] = tableau_colors();
    figure;
    subplot(1, 2, 1);
    if param_id == 1
    	images = {xt2, yt1, xt1, yt2};
    else
    	images = {xr2, yr1, xr1, yr2};
    end
    R = 400;
    S = 68;
    viscircles([0,0], R, 'Color', 'k', 'LineStyle', ':'); hold on;
    plot([-R, R], [0, 0], '--k'); plot([0, 0], [-R, R], '--k');
    xlim([-R-S, R+S]); ylim([-R-S, R+S]); axis equal; axis off;
    z = zeros(68,68,3); z(3:66,3:66,1:3) = template; 
    image(flipud(z), 'XData', [-S S], 'YData', [-S S]); hold on;
    for phase=1:4
        xb = cos((phase-1)*pi/2)*R; 
        yb = sin((phase-1)*pi/2)*R;
        z = zeros(68,68,3); z(3:66,3:66,1:3) = images{phase};
        image(flipud(z), 'XData', [-S+xb S+xb], 'YData', [-S+yb S+yb]);
    end
    scatter(coordinates(:,1)*R/scale_factor, coordinates(:,2)*R/scale_factor, 10, Tableau10(1,:), 'full');
    subplot(1, 2, 2);
    imshow(img);
    print_custom_pdf([out_filename, '.pdf'], [40, 20]);
end
