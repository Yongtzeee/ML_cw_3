function resultNested = SVM()
% load the dataset
data = readtable('online_shoppers_intention_mod.csv');


%% Dataset manipulation for training

% split dataset into features and labels
features = data(:, 1:size(data, 2)-1);
labels = data(:, size(data, 2));

[features, labels] = preProcessData(features, labels);

% split dataset to train and test datasets
featuresTrain = features(1:floor(size(features, 1)/5*4), :);
featuresTest = features(floor(size(features, 1)/5*4)+1:size(features, 1), :);
labelsTrain = labels(1:floor(size(features, 1)/5*4), :);
labelsTest = labels(floor(size(features, 1)/5*4)+1:size(labels, 1), :);

%% (a) Preliminary training to get it working
% train the SVM
modelClassification = fitcsvm(featuresTrain, labelsTrain, 'KernelFunction','linear', 'BoxConstraint',1);
[~, ~, acc] = evaluateSVM(modelClassification, featuresTest, labelsTest);
numSuppVec = size(modelClassification.SupportVectors, 1);
disp("Result from Preliminary training:")
disp("  Accuracy: " + acc * 100)
disp("  Number of Support Vectors: " + numSuppVec)
disp("  Support Vector Ratio: " + numSuppVec / height(features) * 100)


%% (b) Performing inner cross-validation

% hyperparameters
boxConstraints = [0.1, 1, 5, 10, 20];
kernelFunctions = ["linear", "rbf", "polynomial"];
kernelScale = [0.1, 1, 5, 10, 20];
polynomialOrder = 2:4;

% store the results here
resultsNestedCV.linear = zeros(4, 1, length(boxConstraints));
resultsNestedCV.rbf = zeros(4, length(kernelScale), length(boxConstraints));    % number of SV, ratio of SV, accuracy, points
resultsNestedCV.polynomial = zeros(4, length(polynomialOrder), length(boxConstraints));

% take 10% of the original data for faster CV
featuresNestedCV = features(1:height(features)/10, :);
labelsNestedCV = labels(1:height(labels)/10, :);

% train SVM on nested cross-validation
numFoldsNestedCV = 5;
for f = kernelFunctions
        
    count = 1;
    
    for c = boxConstraints
        
        if f == "linear"
            funcArgVals = 1:1;
        elseif f == "rbf"
            funcArgName = 'KernelScale';
            funcArgVals = kernelScale;
        elseif f == "polynomial"
            funcArgName = 'PolynomialOrder';
            funcArgVals = polynomialOrder;
        end
        
        for val = funcArgVals
            
            highestOuterPoint = 0;
            bestOuterModel = 0;
            bestAcc = 0;
            
            for outerFold = 1:numFoldsNestedCV
    
                % split dataset into training and testing datasets in each fold
                featuresFoldTestOuter = featuresNestedCV((outerFold-1)*(floor(size(featuresNestedCV,1)/10))+1:outerFold*(floor(size(featuresNestedCV,1)/10)), :);
                featuresFoldTrain1 = featuresNestedCV(1:(outerFold-1)*(floor(size(featuresNestedCV,1)/10)), :);
                featuresFoldTrain2 = featuresNestedCV(outerFold*(floor(size(featuresNestedCV,1)/10))+1:size(featuresNestedCV,1), :);
                featuresFoldTrainOuter = [featuresFoldTrain1; featuresFoldTrain2];
                labelsFoldTestOuter = labelsNestedCV((outerFold-1)*(floor(size(featuresNestedCV,1)/10))+1:outerFold*(floor(size(featuresNestedCV,1)/10)), :);
                labelsFoldTrain1 = labelsNestedCV(1:(outerFold-1)*(floor(size(featuresNestedCV,1)/10)), :);
                labelsFoldTrain2 = labelsNestedCV(outerFold*(floor(size(featuresNestedCV,1)/10))+1:size(labelsNestedCV,1), :);
                labelsFoldTrainOuter = [labelsFoldTrain1; labelsFoldTrain2];
                
                highestInnerPoint = 0;
                bestInnerModel = 0;

                for innerFold = 1:numFoldsNestedCV

                    % split dataset into training and testing datasets in each fold
                    featuresFoldTestInner = featuresFoldTrainOuter((innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10))+1:innerFold*(floor(size(featuresFoldTrainOuter,1)/10)), :);
                    featuresFoldTrain1 = featuresFoldTrainOuter(1:(innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10)), :);
                    featuresFoldTrain2 = featuresFoldTrainOuter(innerFold*(floor(size(featuresFoldTrainOuter,1)/10))+1:size(featuresFoldTrainOuter,1), :);
                    featuresFoldTrainInner = [featuresFoldTrain1; featuresFoldTrain2];
                    labelsFoldTestInner = labelsFoldTrainOuter((innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10))+1:innerFold*(floor(size(featuresFoldTrainOuter,1)/10)), :);
                    labelsFoldTrain1 = labelsFoldTrainOuter(1:(innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10)), :);
                    labelsFoldTrain2 = labelsFoldTrainOuter(innerFold*(floor(size(featuresFoldTrainOuter,1)/10))+1:size(labelsFoldTrainOuter,1), :);
                    labelsFoldTrainInner = [labelsFoldTrain1; labelsFoldTrain2];
                    
                    % train SVM
                    if f == "linear"
                        modelClassification = fitcsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction',f, 'BoxConstraint',c);
                    else
                        modelClassification = fitcsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction',f, 'BoxConstraint',c, funcArgName,val);
                    end
                    
                    % evaluate SVM
                    [~, ~, acc] = evaluateSVM(modelClassification, featuresFoldTestInner, labelsFoldTestInner);
                    
                    % - a better SVM model is a model that uses less number of support vectors but achieves better accuracy

                    % combine accuracy and SV ratio to calculate "generalization point" for each
                    % best performing hyperparameter combination candidate
                    point = acc * 100 + (100 - size(modelClassification.SupportVectors, 1)/height(featuresFoldTrainInner) * 100);
                    
                    if point > highestInnerPoint
                        bestInnerModel = modelClassification;
                        highestInnerPoint = point;
                    end
                end
                
                % evaluate best performing model
                [~, ~, acc] = evaluateSVM(bestInnerModel, featuresFoldTestOuter, labelsFoldTestOuter);
                
                point = acc * 100 + (100 - size(bestInnerModel.SupportVectors, 1)/height(featuresFoldTrainInner) * 100);
                
                if point > highestOuterPoint
                    bestOuterModel = bestInnerModel;
                    highestOuterPoint = point;
                    bestAcc = acc * 100;
                end
                    
            end
            
            % store number and ratio of support vectors
            numSuppVec = size(bestOuterModel.SupportVectors, 1);
            suppVecRat = numSuppVec / height(featuresFoldTrainOuter) * 100; % in %
            
            if f == "linear"
                resultsNestedCV.linear(count) = numSuppVec;
                resultsNestedCV.linear(count+1) = suppVecRat;
                resultsNestedCV.linear(count+2) = bestAcc;   % in %
                resultsNestedCV.linear(count+3) = highestOuterPoint;
            elseif f == "rbf"
                resultsNestedCV.rbf(count) = numSuppVec;
                resultsNestedCV.rbf(count+1) = suppVecRat;
                resultsNestedCV.rbf(count+2) = bestAcc;   % in %
                resultsNestedCV.rbf(count+3) = highestOuterPoint;
            elseif f == "polynomial"
                resultsNestedCV.polynomial(count) = numSuppVec;
                resultsNestedCV.polynomial(count+1) = suppVecRat;
                resultsNestedCV.polynomial(count+2) = bestAcc;    % in %
                resultsNestedCV.polynomial(count+3) = highestOuterPoint;
            end
            
            count = count + 4;
            
        end
    end
end
resultNested = resultsNestedCV;

bestHyperparamCombi = zeros(2,3);
countHyper = 1;
bc = boxConstraints;
for f = kernelFunctions
    if f == "linear"
        resultMat = resultsNestedCV.linear;
        kfval = nan;
    elseif f == "rbf"
        resultMat = resultsNestedCV.rbf;
        kfval = kernelScale;
    elseif f == "polynomial"
        resultMat = resultsNestedCV.polynomial;
        kfval = polynomialOrder;
    end
    
    % get highest point for each model
    [maxPointsDim2, idx2] = max(resultMat(4,:,:), [], 2);
    [~, idx3] = max(maxPointsDim2, [], 3);
    
    bestHyperparamCombi(countHyper) = kfval(idx2(idx3));
    bestHyperparamCombi(countHyper+1) = bc(idx3);
    
    countHyper = countHyper + 2;
    
    disp("----------------------------------------")
    disp("Best hyperparameter combination for " + f + " kernel function:")
    disp("  Box Constraint: " + bc(idx3))
    disp("  Kernel Function argument value: " + kfval(idx2(idx3)))
    disp("====================")
    disp("Best results from nested cross-validation:")
    disp("  Number of support vectors: " + resultMat(1,idx2(idx3),idx3))
    disp("  Support vector ratio: " + resultMat(2,idx2(idx3),idx3))
    disp("  Accuracy: " + resultMat(3,idx2(idx3),idx3))
end

bestHyperparamCombi = bestHyperparamCombi';

% % display the results from the nested cross-validation
% disp("----------------------------------------")
% disp("Result from Inner Cross-Validation:")
% disp("  Best Accuracy: " + bestAccuracy)
% disp("  Best Number of Support Vectors: " + bestNumSV)
% disp("  Best Support Vector Ratio: " + bestSVRatio)
% disp("Best hyperparameters:")
% disp("  Kernel Function: " + bestKernelFunction)
% disp("  Kernel Function Argument: " + bestFuncArg)
% disp("  Box Constraint: " + bestBoxConstraint)


%% (c1) Perform 10-fold cross-validation for linear, gaussian rbf, and polynomial kernels
SVMPreds = [];
for f = 1:length(kernelFunctions)
    
    totalAcc = 0;
    maxAcc = 0;
    folds = 10;
    for fold = 1:folds
    
        % split dataset into training and testing datasets in each fold
        featuresFoldTest = features((fold-1)*(floor(size(features,1)/10))+1:fold*(floor(size(features,1)/10)), :);
        featuresFoldTrain1 = features(1:(fold-1)*(floor(size(features,1)/10)), :);
        featuresFoldTrain2 = features(fold*(floor(size(features,1)/10))+1:size(features,1), :);
        featuresFoldTrain = [featuresFoldTrain1; featuresFoldTrain2];
        labelsFoldTest = labels((fold-1)*(floor(size(features,1)/10))+1:fold*(floor(size(features,1)/10)), :);
        labelsFoldTrain1 = labels(1:(fold-1)*(floor(size(features,1)/10)), :);
        labelsFoldTrain2 = labels(fold*(floor(size(features,1)/10))+1:size(labels,1), :);
        labelsFoldTrain = [labelsFoldTrain1; labelsFoldTrain2];

        % train SVM
        if kernelFunctions(f) == "linear"
            modelClassification = fitcsvm(featuresFoldTrain, labelsFoldTrain, 'KernelFunction','linear', 'BoxConstraint',bestHyperparamCombi(f,2));
        elseif kernelFunctions(f) == "rbf"
            modelClassification = fitcsvm(featuresFoldTrain, labelsFoldTrain, 'KernelFunction','rbf', 'BoxConstraint',bestHyperparamCombi(f,2), 'KernelScale',bestHyperparamCombi(f,1));
        elseif kernelFunctions(f) == "polynomial"
            modelClassification = fitcsvm(featuresFoldTrain, labelsFoldTrain, 'KernelFunction','polynomial', 'BoxConstraint',bestHyperparamCombi(f,2), 'PolynomialOrder',bestHyperparamCombi(f,1));
        end

        % evaluate SVM
        [~, ~, acc] = evaluateSVM(modelClassification, featuresFoldTest, labelsFoldTest);

        % average and max accuracy
        if acc > maxAcc
            maxAcc = acc;
        end
        totalAcc = totalAcc + acc;
        
    end
    
    avgAcc = totalAcc / folds * 100;
    disp("----------------------------------------")
    disp("Result for " + kernelFunctions(f) + " kernel function in 10-fold cross-validation:")
    disp("  Max accuracy: " + maxAcc * 100)
    disp("  Average accuracy: " + avgAcc)
    
    % train and evaluate best SVM model on whole dataset and retrieve its predictions
    if kernelFunctions(f) == "linear"
        modelClassification = fitcsvm(featuresTrain, labelsTrain, 'KernelFunction','linear', 'BoxConstraint',bestHyperparamCombi(f,2));
    elseif kernelFunctions(f) == "rbf"
        modelClassification = fitcsvm(featuresTrain, labelsTrain, 'KernelFunction','rbf', 'BoxConstraint',bestHyperparamCombi(f,2), 'KernelScale',bestHyperparamCombi(f,1));
    elseif kernelFunctions(f) == "polynomial"
        modelClassification = fitcsvm(featuresTrain, labelsTrain, 'KernelFunction','polynomial', 'BoxConstraint',bestHyperparamCombi(f,2), 'PolynomialOrder',bestHyperparamCombi(f,1));
    end
    
    % evaluate best model for each kernel and store their results and predictions
    [preds, ~, acc] = evaluateSVM(modelClassification, featuresTest, labelsTest);
    numSuppVec = size(modelClassification.SupportVectors, 1);
    disp("----------------------------------------")
    disp("Result from cross-validation training:")
    disp("  Accuracy: " + acc * 100)
    disp("  Number of Support Vectors: " + numSuppVec)
    disp("  Support Vector Ratio: " + numSuppVec / height(features) * 100)
    
    SVMPreds = [SVMPreds preds];
    
end


%% (c2) compare results between ANN, Decision Tree, and SVM
% get predictions for Decision Tree and ANN
ANNPreds = [1 0 0 0 1 1 1 0 0 0 1 0 1 0 0 0 1 0 0 1 0 1 0 1 0 1 0 1 1 1 1 0 1 1 0 1 0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 1 0 1 1 0 0 1 0 0 0 1 0 0 1 0 1 0 1 0 0 1 1 1 0 1 1 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0 1 1 1 0 1 0 1 1 0 0 1 1 1 0 1 1 1 1 0 1 1 0 0 1 0 0 1 1 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 1 1 0 1 0 0 0 1 1 1 1 1 0 0 1 0 1 0 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 0 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 0 0 1 1 1 0 0 1 1 1 0 0 1 1 1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 1 0 0 0 1 1 0 0 1 1 0 1 1 1 0 0 1 0 0 1 0 1 1 1 0 1 1 0 1 1 0 0 1 1 1 0 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 0 1 1 0 0 1 0 1 1 1 0 1 0 1 1 1 1 0 0 1 1 0 0 1 0 1 1 1 0 1 0 1 1 0 0 1 1 1 1 1 0 1 1 1 0 1 1 0 1 0 0 0 1 1 0 1 0 0 0 0 1 0 1 1 1 1 1 1 0 0 1 1 0 0 1 1 0 1 0 1 0 0 0 0 1 1 1 0 1 1 0 0 0 0 0 1 1 1 0 0 1 0 0 1 0 0 1 0 1 0 1 1 1 0 0 1 0 1 1 1 0 0 0 1 1 1 1 0 1 0 1 0 1 1 1 0 1 0 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0 0 1 0 0 1 1 1 1 0 1 1 1 0 0 1 1 1 1 1 1 1 1 0 0 0 1 1 1 0 1 1 0 1 0 0 1 1 1 0 0 0 1 1 0 0 0 0 1 1 0 0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 0 0 0 1 1 0 1 0 1 0 1 0 0 0 0 1 1 0 0 1 1 0 1 1 1 0 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 0 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 0 0 0 1 1 0 1 0 0 1 1 1 1 0 1 0 1 1 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 1 0 1 1 1 0 0 0 0 0 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 1 1 1 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 0 1 0 1 1 0 1 1 1 0 1 1 1 1 1]';
DTPreds = [0 0 0 0 0 1 1 0 0 0 1 0 1 1 0 0 1 0 0 1 1 1 0 1 0 1 1 1 1 0 1 0 0 0 0 1 0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 1 1 0 1 0 0 0 1 1 1 1 1 0 1 1 0 0 1 1 0 0 1 1 0 0 0 1 0 1 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 1 0 0 1 1 0 1 1 1 1 1 0 0 1 0 0 1 1 0 1 1 0 0 1 1 0 1 1 1 1 0 0 1 0 0 0 0 1 1 0 1 0 0 0 0 1 1 0 1 0 0 1 1 1 0 0 0 0 1 1 0 1 1 0 1 1 1 1 1 0 1 0 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 0 0 1 1 1 1 1 0 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 0 1 1 1 1 1 0 1 0 0 1 1 1 1 0 1 1 1 1 0 1 1 0 0 1 1 0 0 1 1 1 0 1 1 1 1 1 0 1 0 0 1 1 0 0 0 1 0 1 1 1 0 0 1 0 0 1 1 1 1 1 0 1 1 0 1 1 0 0 1 1 1 0 1 0 1 0 1 0 1 1 1 0 1 0 1 1 1 0 1 1 1 0 1 0 1 1 1 0 1 1 1 0 1 1 0 0 1 1 0 1 1 0 1 1 0 0 1 0 1 1 0 0 1 1 1 1 1 0 1 1 1 0 1 1 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 1 1 1 0 0 1 1 0 0 0 1 1 0 1 0 1 1 1 0 1 1 0 1 0 0 0 1 0 1 1 1 1 0 1 0 1 0 1 1 1 0 0 0 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 0 0 1 1 1 1 0 0 0 1 0 1 0 0 0 1 0 1 1 0 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 0 0 1 1 0 1 1 0 1 0 1 1 1 1 0 0 0 1 1 0 0 0 0 1 1 0 0 1 0 0 1 1 1 0 0 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 0 0 0 1 1 1 0 0 0 1 1 1 0 0 1 1 0 1 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 1 1 0 0 1 1 1 0 0 0 1 0 0 1 0 0 1 1 1 1 0 1 0 1 1 1 0 1 0 0 0 1 0 0 0 1 1 0 0 1 1 0 1 1 1 1 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 1 0 1 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 1 0 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 0 1 1 0 0 1 0 1 1 0 0 1 1 0 1 1 1 1 1]';
SVMLinearPreds = SVMPreds(:,1);
SVMRBFPreds = SVMPreds(:,2);
SVMPolynomialPreds = SVMPreds(:,3);

modelPreds = [ANNPreds DTPreds SVMLinearPreds SVMRBFPreds SVMPolynomialPreds];
models = ["ANN" "DecisionTree" "SVMLinear" "SVMRBF" "EVMPolynomial"];
numModels = length(models);
count = 1;
pairing = [];
ttest2Results = zeros(3,10);    % 4+3+2+1 pairings

% calculate accuracy of all the predictions
disp("----------------------------------------")
disp("Accuracy of each model:")
accuracyOfModels = zeros(1,length(models));
for i = 1:length(models)
    
    pred = modelPreds(:, i);
    acc = calculateAccuracy(pred, table2array(labelsTest));
    accuracyOfModels(i) = acc;
    
    disp(models(i) + ": " + acc)
end

% run ttest on all combinations
disp("----------------------------------------")
disp("TTEST2 Results:")
for i = 1:numModels-1
    for j = i+1:numModels
        
        pairing = [pairing; models(i)+" + "+models(j)];
        
        [h,p,~,stats] = ttest2(modelPreds(:,i), modelPreds(:,j));
        ttest2Results(count) = h;
        ttest2Results(count+1) = p;
        ttest2Results(count+2) = stats.tstat;
        
        count = count + 3;
        
        disp(models(i) + " + " + models(j))
        disp("  h: " + h + ", p: " + p + ", t-stat: " + stats.tstat)
    end
end

ttest2Results = ttest2Results';

end


%% calculate accuracy
function accuracy = calculateAccuracy(predicted, actual)

totalCorrect = 0;
for p = 1:length(predicted)
    if predicted(p) == actual(p)
        totalCorrect = totalCorrect + 1;
    end
end

accuracy = totalCorrect / length(predicted);
    
end


%% evaluate SVM function
function [preds, scores, acc] = evaluateSVM(model, features, labels)

[preds, scores] = predict(model, features);

labs = table2array(labels);

acc = calculateAccuracy(preds, labs);

end


%% pre-process the dataset
function [features, labels] = preProcessData(features, labels)

% remove irrelevant attributes
features(:, {'Administrative' 'Informational' 'ProductRelated' 'OperatingSystems' 'Browser'}) = [];

% process features
features.Weekend = findgroups(features.Weekend) - 1;
features.VisitorType = findgroups(features.VisitorType) - 1;
features.Month = findgroups(features.Month) - 1;

% process labels
labels.Revenue = findgroups(labels.Revenue) - 1;

% normalize dataset
features = normalize(features);

end


