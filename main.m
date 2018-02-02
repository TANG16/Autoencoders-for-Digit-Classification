

% Load the training data into memory
%[xTrainImages,tTrain] = digitTrainCellArrayData;

% Load the test images
%[xTestImages,tTest] = digitTestCellArrayData;


[xTrainImages,tTrain, xTestImages, tTest] = mnist_data();
% Display some of the training images
%{
clf
for i = 1:20
    subplot(4,5,i);
    imshow(xTrainImages{i});
end
%}


%% seeding the random number generator
rng('default')

%% hidden layer 1
%% hidden layer size (smaller than the input layer)
hiddenSize1 = 100;

autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

view(autoenc1);

feat1 = encode(autoenc1,xTrainImages);

%% hidden layer 2
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

view(autoenc2);
feat2 = encode(autoenc2,feat1);


%% hidden layer 3
%{
hiddenSize3 = 25;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

view(autoenc3);
feat3 = encode(autoenc3,feat2);
%}


softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);
deepnet = stack(autoenc1,autoenc2,softnet);
%softnet = trainSoftmaxLayer(feat3,tTrain,'MaxEpochs',400);
%deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);


% Get the number of pixels in each image
imageWidth = size(xTestImages{1},1);
imageHeight = size(xTestImages{1},2);
inputSize = imageWidth*imageHeight;


% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end

y = deepnet(xTest);
figure(10);
plotconfusion(tTest,y);

% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end

% Perform fine tuning
deepnet = train(deepnet,xTrain,tTrain);


y = deepnet(xTest);
figure(11);
title('finetuned');
plotconfusion(tTest,y);

%nnet.guis.closeAllViews() 
