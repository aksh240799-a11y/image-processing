%% Confusion Matrix %%

load /CourseData/aslDetectorMaxEp100.mat aslDetector 
load groundTruthVars.mat gtTest gtData
testSet = gtTest;
results = detect(aslDetector,testSet)

metrics = evaluateObjectDetection(results,testSet,0.85)
[confMat,confLabels] = confusionMatrix(metrics)
confusionchart(confMat{1},confLabels)


%% Precision and Recall %%

load /CourseData/aslDetectorMaxEp100.mat aslDetector
load /CourseData/aslVowels.mat gTruthVowels
[imds,bxds] = objectDetectorTrainingData(gTruthVowels);
gtData = combine(imds,bxds);
results = detect(aslDetector,gtData,Threshold=0.05);
metrics = evaluateObjectDetection(results,gtData);

[p,r,s] = precisionRecall(metrics)
plot(s{4},p{4},"o-")
xlabel("Score")
ylabel("Precision")
title("O")

plot(s{4},r{4},"o-")
xlabel("Score")
ylabel("Recall")
title("O")

plot(r{4},p{4},"o-")
xlabel("Recall")
ylabel("Precision")
title("O")


%% Detection Score Threshold Based on Precision-Recall Curves %%

load /CourseData/aslDetectorMaxEp100.mat aslDetector
load groundTruthVar.mat gtData
load prsVariables.mat p r s

plot(s{1},p{1},"o-")
hold on
plot(s{2},p{2},"o-")
plot(s{3},p{3},"o-")
plot(s{4},p{4},"o-")
plot(s{5},p{5},"o-")
hold off
xline(0.4)
xlabel("Score")
ylabel("Precision")
legend("A","E","I","O","U",Location="southeast")
grid on

plot(s{1},r{1},"o-")
hold on
plot(s{2},r{2},"o-")
plot(s{3},r{3},"o-")
plot(s{4},r{4},"o-")
plot(s{5},r{5},"o-")
hold off
xline(0.4)
xlabel("Score")
ylabel("Recall")
legend("A","E","I","O","U",Location="southwest")
grid on

plot(r{1},p{1},"o-")
hold on
plot(r{2},p{2},"o-")
plot(r{3},p{3},"o-")
plot(r{4},p{4},"o-")
plot(r{5},p{5},"o-")
hold off
xlabel("Recall")
ylabel("Precision")
legend("A","E","I","O","U",Location="southwest")
grid on

results = detect(aslDetector,gtData,Threshold=0.4);
metrics = evaluateObjectDetection(results,gtData);
[confMat,confLabels] = confusionMatrix(metrics);
confusionchart(confMat{1},confLabels)
