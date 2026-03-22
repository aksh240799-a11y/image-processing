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

