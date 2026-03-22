%% Define Hpyerparameters %%

load validationData.mat gtVal
detector = yoloxObjectDetector("tiny-coco", ...
    ["A" "E" "I" "O" "U"], ...
    InputSize=[224 224 3])
   options = trainingOptions("adam", ...
    MiniBatchSize=70, ...      % 2 iterations per epoch
    MaxEpochs=10, ...
    ValidationData=gtVal, ...
    ValidationFrequency=2, ... % need to specify or it is every 50 iterations
    Plots="training-progress") % shows up after 1 epoch




%% Train a Detector %% 
% create ground truth datastore
load /CourseData/aslVowels.mat gTruthVowels
[imds,bxds] = objectDetectorTrainingData(gTruthVowels);
gtData = combine(imds,bxds);

% split data into training, validation, and testing sets
gtShuffled = shuffle(gtData);
numFiles = numpartitions(gtShuffled);
numTraining = floor(numFiles*0.7);
gtTrain = subset(gtShuffled,1:numTraining);
numValidation = floor(numFiles*0.1);
gtVal = subset(gtShuffled,(numTraining+1):(numTraining+numValidation));

% create small, toy dataset
gtToy = subset(gtTrain,1:10)

% define variables required for training
detector = yoloxObjectDetector("tiny-coco", ...
    ["A" "E" "I" "O" "U"], ...
    InputSize=[224 224 3]);
optionsToy = trainingOptions("adam", ...
    MaxEpochs=1);
options = trainingOptions("adam", ...
    MiniBatchSize=70, ...      % 2 iterations per epoch
    MaxEpochs=10, ...
    ValidationData=gtVal, ...
    ValidationFrequency=2, ... % need to specify or it is every 50 iterations
    Plots="training-progress") % shows up after 1 epoch


aslDetectorToy = trainYOLOXObjectDetector(gtToy,detector,optionsToy);




%% Detecting Object using PreTrained Detector %%
load /CourseData/aslDetector_allTrainingData10Epochs.mat
manA = imread("manSigningA.jpg");

load gtTestData gtTest
[dbox,dscore,dlabel] = detect(aslDetector,manA,Threshold=0.75)
detectedIm = insertObjectAnnotation(manA,"rectangle",dbox,dlabel);
imshow(detectedIm)
title("The Letter A")

results = detect(aslDetector,gtTest)
