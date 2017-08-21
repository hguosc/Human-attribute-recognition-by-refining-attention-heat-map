function crops_data = prepare_image_vgg(im)
% ------------------------------------------------------------------------
IMAGE_DIM = 256;
CROPPED_DIM = 224; % 224 for googLeNet , 227 for VGG and AlexNet
mean_data = zeros(IMAGE_DIM, IMAGE_DIM, 3);
mean_data(:,:,1) = 104;     %B
mean_data(:,:,2) = 117;     %G
mean_data(:,:,3) = 123;     %R

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)
crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 1, 'single');
crops_data(:,:,:,1) = imresize(im_data, [CROPPED_DIM CROPPED_DIM], 'bilinear');