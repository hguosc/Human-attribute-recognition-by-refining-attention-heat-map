% Sample code to generate the heat map for an image.
% Based on Bolei Zhou's work: https://github.com/metalbubble/CAM
% This sample code just work for VGG16-based models

clear
close all

%% Set up
addpath('./caffe/matlab');
str = '03533.jpg';
net_weights = 'weights for a VGG16 based model';
net_model = 'a VGG16 based model';
attr = 2;   % select an attribute: 1 ~ 9
attributes = {'male', 'long hair', 'glasses', 'hat', 'tshirt', 'long sleeves', 'short', 'jeans', 'long pants'};

categories = {'no', 'yes'};
img = imread(str);
[row_orig col_orig cha_orig] = size(img);
img = imresize(img, [256 256]);

%% Forward the image through the network
net = caffe.Net(net_model, net_weights, 'test');    

fc = ['cam_fc_a' num2str(attr)];
weights_LR = net.params(fc,1).get_data();% get the softmax layer of the network

input_data = prepare_image_vgg(img);
input_data2 = imresize(input_data, [220, 220]);

scores = net.forward({input_data});
activation_lastconv = net.blobs('cam_conv').get_data();
scores = scores{attr}; 

blob_str = ['reshape_ave_a' num2str(attr)];
heatmap = net.blobs(blob_str).get_data();
heatmap = abs(double(heatmap));

ratio = heatmap./(sum(sum(heatmap)));

heat_max = max(max(heatmap));
heat_ave1 = sum(sum(heatmap))/(14*14);
thresh = heat_max*0.5;
heatmap(heatmap < thresh) = 0;


heat_ave2 = sum(sum(heatmap))/(14*14);
heatmap_sig = sigmf(heatmap, [0.1,0]);
heatmap_sig = heatmap_sig.*2 - 1;
heatmap_sig = imresize(heatmap_sig, [224, 224]);

d3(:,:,1) = heatmap_sig;
d3(:,:,2) = heatmap_sig;
d3(:,:,3) = heatmap_sig;

prod_img = double(input_data).*d3;
prod_img(:,:,1) = prod_img(:,:,1) + 104;
prod_img(:,:,2) = prod_img(:,:,2) + 117;
prod_img(:,:,3) = prod_img(:,:,3) + 123;
prod_img = prod_img(:,:,[3,2,1]);
prod_img = permute(prod_img, [2,1,3]);
    

%% Class Activation Mapping

topNum = 2; % generate heatmap for top X prediction results
scoresMean = mean(scores,2);
[value_category, IDX_category] = sort(scoresMean,'descend');
[curCAMmapAll] = returnCAMmap(activation_lastconv, weights_LR(:,IDX_category(1:topNum)));

curCAMmapAll = abs(curCAMmapAll);
curCAMmapAll = permute(curCAMmapAll, [2,1,3]);

curResult = im2double(img);
curPrediction = '';

for j=1:topNum
    curCAMmap_crops = squeeze(curCAMmapAll(:,:,j,:));
    curCAMmapLarge_crops = imresize(curCAMmap_crops,[256 256]);

    curCAMLarge = curCAMmapLarge_crops;
    curHeatMap = imresize(im2double(curCAMLarge),[256 256]);
    curHeatMap = im2double(curHeatMap);

    curHeatMap = map2jpg(curHeatMap,[], 'jet');
    curHeatMap = im2double(img)*0.5+curHeatMap*0.5;
    
    curResult = [curResult ones(size(curHeatMap,1),8,3) curHeatMap];
    curPrediction = [curPrediction ' --rank'  num2str(j) ':' categories{IDX_category(j)} '    '];
end

figure
curPrediction = sprintf('%s? %s', attributes{attr}, curPrediction);
imshow(curResult);title(curPrediction)
figure
imshow(uint8(prod_img));title('Positive Attention Area')

caffe.reset_all();


