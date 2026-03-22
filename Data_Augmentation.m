%% Augment Images and Bounding Boxes %%

load /CourseData/aslVowels.mat gTruthVowels
[imds,bxds] = objectDetectorTrainingData(gTruthVowels);
gtData = combine(imds,bxds);

data = read(gtData)
im = data{1};
bbox = data{2}
label = data{3}
annotatedImage = insertObjectAnnotation(im,"rectangle",bbox,label);
imshow(annotatedImage)

tform = randomAffine2d(XReflection=true)
augmentedImage = imwarp(im,tform);
imshow(augmentedImage)
spatialRef = affineOutputView(size(augmentedImage),tform)
augmentedBoxes = bboxwarp(bbox,tform,spatialRef);


%% Data Augmentation Function %%
load /CourseData/aslVowels.mat gTruthVowels
[imds,bxds] = objectDetectorTrainingData(gTruthVowels);
gtData = combine(imds,bxds);
data = read(gtData);
annotatedImage = insertObjectAnnotation(data{1},"rectangle",data{2},data{3});
imshow(annotatedImage)

function out = flipAug(data)

% parse data read from datastore
im = data{1};
bbox = data{2};
label = data{3};
% create transform that flips images horizontally
tform = randomAffine2d(XReflection=true);
% warp image with random flip
augmentedImage = imwarp(im,tform);
% create spacial refernce for transform
spatialRef = affineOutputView(size(augmentedImage),tform);
% warp bounding boxes with random flip
augmentedBoxes = bboxwarp(bbox,tform,spatialRef);

% TASK 1: combine images, bounding boxes and labels
out = {augmentedImage augmentedBoxes label};

end
dataAug = flipAug(data);

%% Transform Ground Truth Data with Data Augmentation %%
% prepare ground truth data
load /CourseData/aslVowels.mat gTruthVowels
[imds,bxds] = objectDetectorTrainingData(gTruthVowels);
gtData = combine(imds,bxds);

% split images into training, validation, and testing
gtShuffled = shuffle(gtData);
numFiles = numpartitions(gtShuffled);
numTraining = floor(numFiles*0.7);
gtTrain = subset(gtShuffled, 1:numTraining);
numValidation = floor(numFiles*0.1);
gtVal = subset(gtShuffled, (numTraining+1):(numTraining + numValidation));
dataAug = read(gtTrainAug)
imAug = dataAug{1};
bboxAug = dataAug{2};
label = dataAug{3};
annotatedImageAug = insertObjectAnnotation(imAug,"rectangle",bboxAug,label);
imshow(annotatedImageAug)
gtValAug = transform(gtVal,@flipAug)

%% Evaluate effictiveness of Data Augmentation %%
load /CourseData/aslDetectorMaxEp100 aslDetector
load /CourseData/aslDetectorMaxEp100_withAugmentation aslDetector_withAugmentation 
load groundTruthVars.mat gtData gtTest gTruthVowels gtLeftSigning gtTest_LeftSigning
detectorToEvaluate = aslDetector_withAugmentation;
groundTruthToEvaluate = gtTest_LeftSigning;
results = detect(detectorToEvaluate,groundTruthToEvaluate, ...
    Threshold=0.4);
metrics = evaluateObjectDetection(results, ...
    groundTruthToEvaluate);
[confMat,confLabels] = confusionMatrix(metrics);
confusionchart(confMat{1},confLabels)



