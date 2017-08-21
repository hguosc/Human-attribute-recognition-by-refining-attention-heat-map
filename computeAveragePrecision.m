% Sample code to calculate the average precision of trained model. Make
% sure the matcaffe has been compiled. The test is based on the testset of
% Berkeley Attribute of Human People Dataset. The test images have been
% cropped based on the annotated bounding boxes.


% set up
net_type = 1;   % 1: AlexNet based; 2: VGG16 based

addpath('./caffe/matlab');      % specify the caffe path
run('./vlfeat-0.9.20/toolbox/vl_setup.m');      % specify the vl_feat path

test_gt = './test_list.txt';    % path to test_list GT. The test images are pre-cropped based on GT.
[imgs, lab1, lab2, lab3, lab4, lab5, lab6, lab7, lab8, lab9] = ...
    textread(test_gt, '%s%d%d%d%d%d%d%d%d%d');
model = './models/deploy_AlexNet_CAM.prototxt';     % the model needed to be tested
weights = './models/AlexNet_CAM_Refined.caffemodel' % the weights of the model

tic

caffe.set_mode_gpu();
pause(3);
net = caffe.Net(model, weights, 'test');
if net_type == 1
    crop_size = 227;
else
    crop_size = 224;
end

mean_data = zeros(crop_size, crop_size, 3);
mean_data(:,:,1) = 104;     %B
mean_data(:,:,2) = 117;     %G
mean_data(:,:,3) = 123;     %R
mean_data = single(mean_data);

% prob_all
attr_num = 9;
prob_all = cell(attr_num, numel(imgs));
for i =1:numel(imgs)
    im = imread(imgs{i});
    im = imresize(im, [crop_size, crop_size]);
    im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = im_data - mean_data;
    prob = net.forward({im_data});
    prob_all(:,i) = prob(1:attr_num);
end

toc
% labels_all
labels_all = [lab1'; lab2'; lab3'; lab4'; lab5'; lab6'; lab7'; lab8'; lab9'];

attributes = {'male', 'long hair', 'glasses', 'hat', 'tshirt', 'long sleeves', 'short', 'jeans', 'long pants'};
figure
sum_all = 0;
display(weights)
for attr = 1:attr_num
    attr_label = labels_all(attr,:);
    attr_prob = prob_all(attr,:);
    index_pos = find(attr_label == 1);
    index_neg = find(attr_label == -1);
    index_non = find(attr_label == 0);
    prob = prob_all(attr,:);
    prob1 = [];
    prob2 = [];
    prob3 = [];
    for i = 1:numel(prob)
        prob1(i) = prob{i}(1);
        prob2(i) = prob{i}(2);
    end

    h = subplot(3,3,attr);
    title(['attribute ' num2str(attr)])  
    pos_label = labels_all(attr,:);
    pos_label(index_non) = [];
    prob2(index_non) = [];
    vl_pr(pos_label, prob2);
    str = get(h, 'Title');
    c = textscan(str.String, '%s %s %f %s %s %f %s %s %f %s');
    sum_all = sum_all + c{6};
    display(['attr: ' num2str(attr) '; ap = ' num2str(c{6})]);
    t = sprintf('attribute: %s\n%s', attributes{attr}, str.String);
    title(t);
end
display(['ap = ' num2str(sum_all / 9)]);
caffe.reset_all;
