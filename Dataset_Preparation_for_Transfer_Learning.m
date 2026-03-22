//Ground Truth Data//

load /CourseData/aslVowels.mat gTruthVowels
gTruthVowels
gTruthVowels.LabelData

[imds,bxds] = objectDetectorTrainingData(gTruthVowels);
cts = countEachLabel(bxds)
gtData = combine(imds,bxds);
data = read(gtData)
im = data{1};
bb = data{2}
lab = data{3}

imLabeled = insertObjectAnnotation( ...
    im,"rectangle",bb,lab);
imshow(imLabeled)

// Splitting the dataset - Train, Validation and Test Data//
load /CourseData/aslVowels.mat gTruthVowels
[imds,bxds] = objectDetectorTrainingData(gTruthVowels);
gtData = combine(imds,bxds);
countEachLabel(bxds)
gtShuffled = shuffle(gtData);

numFiles = numpartitions(gtShuffled)
numTrain = floor(numFiles*0.7)
numVal = floor(numFiles*0.1)

gtTrain = subset(gtShuffled,1:numTrain);
gtVal = subset(gtShuffled,(numTrain+1):(numTrain+numVal));
gtTest = subset(gtShuffled,(numTrain+numVal+1):numFiles);
bxTrain = gtTrain.UnderlyingDatastores{2}
trainCounts = countEachLabel(bxTrain)
