function finalResult = SVM()
% load the dataset
data = readtable('online_shoppers_intention_mod.csv');


%% Dataset manipulation for training

% split dataset into features and labels
features = data(:, 1:size(data, 2)-1);
labels = data(:, size(data, 2));

[features, labels] = preProcessData(features, labels);


% %% (a) Preliminary training to get it working
% % train the SVM
% modelClassification = fitcsvm(features, labels, 'KernelFunction','linear', 'BoxConstraint',1);
% [~, ~, acc] = evaluateSVM(modelClassification, features, labels);
% numSuppVec = size(modelClassification.SupportVectors, 1);
% disp("Result from Preliminary training:")
% disp("  Accuracy: " + acc * 100)
% disp("  Number of Support Vectors: " + numSuppVec)
% disp("  Support Vector Ratio: " + numSuppVec / height(features) * 100)


%% (b) Performing inner cross-validation

% hyperparameters
boxConstraints = [0.001, 0.1, 1, 10, 100];
kernelFunctions = ["rbf", "polynomial"];
kernelScale = [0.001, 0.1, 1, 10, 100];
polynomialOrder = 2:4;

% store the results here
results.rbf = zeros(3, length(kernelScale), length(boxConstraints));    % number of SV, ratio of SV, accuracy
results.polynomial = zeros(3, length(polynomialOrder), length(boxConstraints));

% take 10% of the original data for faster CV
featuresNestedCV = features(1:height(features)/10, :);
labelsNestedCV = labels(1:height(labels)/10, :);

% train SVM on nested cross-validation
numFoldsNestedCV = 5;
for f = kernelFunctions
        
    count = 1;
    
    for c = boxConstraints
        
        if f == "rbf"
            funcArgName = 'KernelScale';
            funcArgVals = kernelScale;
        elseif f == "polynomial"
            funcArgName = 'PolynomialOrder';
            funcArgVals = polynomialOrder;
        end
        
        for val = funcArgVals
            
            highestOuterAcc = 0;
            bestOuterModel = 0;
            
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
                
                highestInnerAcc = 0;
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
                    
                    tic
                    modelClassification = fitcsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction',f, 'BoxConstraint',c, funcArgName,val);
                    toc
                    
                    % evaluate
                    [~, ~, acc] = evaluateSVM(modelClassification, featuresFoldTestInner, labelsFoldTestInner);
                    
                    if acc > highestInnerAcc
                        bestInnerModel = modelClassification;
                        highestInnerAcc = acc;
                    end
                end
                
                % evaluate best performing model
                [~, ~, acc] = evaluateSVM(bestInnerModel, featuresFoldTestOuter, labelsFoldTestOuter);
                
                if acc > highestOuterAcc
                    bestOuterModel = modelClassification;
                    highestOuterAcc = acc;
                end
                    
            end
            
            % store number and ratio of support vectors
            numSuppVec = size(bestOuterModel.SupportVectors, 1);
            suppVecRat = numSuppVec / height(featuresFoldTrainOuter) * 100; % in %
            
            if f == "rbf"
                results.rbf(count) = numSuppVec;
                results.rbf(count+1) = suppVecRat;
                results.rbf(count+2) = highestOuterAcc * 100;   % in %
            elseif f == "polynomial"
                results.polynomial(count) = numSuppVec;
                results.polynomial(count+1) = suppVecRat;
                results.polynomial(count+2) = highestOuterAcc * 100;    % in %
            end
            
            count = count + 3;
            
        end
    end
end
finalResult = results;

% get best hyperparameter combination candidates
% accuracy and SV ratio of best accuracy in RBF
[maxAccRBF1, idxAccRBF1] = max(results.rbf(3,:,:), [], 2);
[bestRBFAcc, idxAccRBF2] = max(maxAccRBF1, [], 3);
bestAccSVRatRBF = results.rbf(2,idxAccRBF1(idxAccRBF2),idxAccRBF2);

% accuracy and SV ratio of best SV ratio in RBF
[minSVRatRBF1, idxRatRBF1] = min(results.rbf(2,:,:), [], 2);
[bestRBFSVRat, idxRatRBF2] = min(minSVRatRBF1, [], 3);
bestSVRatAccRBF = results.rbf(3,idxRatRBF1(idxRatRBF2),idxRatRBF2);

% accuracy and SV ratio of best accuracy in Polynomial
[maxAccPoly1, idxAccPoly1] = max(results.polynomial(3,:,:), [], 2);
[bestPolyAcc, idxAccPoly2] = max(maxAccPoly1, [], 3);
bestAccSVRatPoly = results.polynomial(2,idxAccPoly1(idxAccPoly2),idxAccPoly2);

% accuracy and SV ratio of best SV ratio in Polynomial
[minSVRatPoly1, idxRatPoly1] = min(results.polynomial(2,:,:), [], 2);
[bestPolySVRat, idxRatPoly2] = min(minSVRatPoly1, [], 3);
bestSVRatAccPoly = results.polynomial(3,idxRatPoly1(idxRatPoly2),idxRatPoly2);

% - a better SVM model is a model that uses less number of support vectors but achieves better accuracy

% combine accuracy and SV ratio to calculate "generalization point" for each
% best performing hyperparameter combination candidate
accuracies = [bestRBFAcc bestSVRatAccRBF bestPolyAcc bestSVRatAccPoly];
ratio = [bestAccSVRatRBF bestRBFSVRat bestAccSVRatPoly bestPolySVRat];
ratioInv = 100 - ratio;
points = accuracies + ratioInv;
[~, idx] = max(points);

bestAccuracy = accuracies(idx);
bestSVRatio = ratio(idx);

% get statistics of best performing SVM model from the nested
% cross-validation
switch(idx)
    case 1
        bestKernelFunction = "rbf";
        bestFuncArg = kernelScale(idxAccRBF1(idxAccRBF2));
        bestBoxConstraint = boxConstraints(idxAccRBF2);
        bestNumSV = results.rbf(1,idxAccRBF1(idxAccRBF2),idxAccRBF2);
    case 2
        bestKernelFunction = "rbf";
        bestFuncArg = kernelScale(idxRatRBF1(idxRatRBF2));
        bestBoxConstraint = boxConstraints(idxRatRBF2);
        bestNumSV = results.rbf(1,idxRatRBF1(idxRatRBF2),idxRatRBF2);
    case 3
        bestKernelFunction = "polynomial";
        bestFuncArg = polynomialOrder(idxAccPoly1(idxAccPoly2));
        bestBoxConstraint = boxConstraints(idxAccPoly2);
        bestNumSV = results.polynomial(1,idxAccPoly1(idxAccPoly2),idxAccPoly2);
    case 4
        bestKernelFunction = "polynomial";
        bestFuncArg = polynomialOrder(idxRatPoly1(idxRatPoly2));
        bestBoxConstraint = boxConstraints(idxRatPoly2);
        bestNumSV = results.polynomial(1,idxRatPoly1(idxRatPoly2),idxRatPoly2);
    otherwise
        bestKernelFunction = "NONE";
        bestFuncArg = -1;
        bestBoxConstraint = -1;
        bestNumSV = -1;
end

% display the results from the nested cross-validation
disp("________________________________________")
disp("Result from Inner Cross-Validation:")
disp("  Best Accuracy: " + bestAccuracy)
disp("  Best Number of Support Vectors: " + bestNumSV)
disp("  Best Support Vector Ratio: " + bestSVRatio)
disp("Best hyperparameters:")
disp("  Kernel Function: " + bestKernelFunction)
disp("  Kernel Function Argument: " + bestFuncArg)
disp("  Box Constraint: " + bestBoxConstraint)


% %% (c1) Perform 10-fold cross-validation
% folds = 10;
% for fold = 1:folds
%     
%     % split dataset into training and testing datasets in each fold
%     featuresFoldTest = features((fold-1)*(floor(size(features,1)/10))+1:fold*(floor(size(features,1)/10)), :);
%     featuresFoldTrain1 = features(1:(fold-1)*(floor(size(features,1)/10)), :);
%     featuresFoldTrain2 = features(fold*(floor(size(features,1)/10))+1:size(features,1), :);
%     featuresFoldTrain = [featuresFoldTrain1; featuresFoldTrain2];
%     labelsFoldTest = labels((fold-1)*(floor(size(features,1)/10))+1:fold*(floor(size(features,1)/10)), :);
%     labelsFoldTrain1 = labels(1:(fold-1)*(floor(size(features,1)/10)), :);
%     labelsFoldTrain2 = labels(fold*(floor(size(features,1)/10))+1:size(labels,1), :);
%     labelsFoldTrain = [labelsFoldTrain1; labelsFoldTrain2];
%     
%     % train and evaluate
%     
%     
% end

%% (c2) compare results between ANN, Decision Tree, and SVM



end


%% evaluate SVM function
function [preds, scores, acc] = evaluateSVM(model, features, labels)

[preds, scores] = predict(model, features);

labs = table2array(labels);

totalCorrect = 0;
for i = 1:length(preds)
    if preds(i) == labs(i)
        totalCorrect = totalCorrect + 1;
    end
end

acc = totalCorrect / length(preds);

end


%% pre-process the dataset
function [features, labels] = preProcessData(features, labels)

% remove irrelevant attributes
features(:, {'Administrative' 'Informational' 'ProductRelated' 'OperatingSystems' 'Browser'}) = [];

% % rename table column names for better display of decision tree
% features.Properties.VariableNames = {'AdmDu', 'InfoDu', 'ProdDu', 'BounceR', 'ExitR', 'PageV', 'SpecD', 'Month', 'Region', 'TrafficT', 'VisitorT', 'Weekend'};

% process features
features.Weekend = findgroups(features.Weekend) - 1;
features.VisitorType = findgroups(features.VisitorType) - 1;
features.Month = findgroups(features.Month) - 1;

% process labels
labels.Revenue = findgroups(labels.Revenue) - 1;

% normalize dataset
features = normalize(features);

end


