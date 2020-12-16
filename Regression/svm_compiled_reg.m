%% Load data
data_red = readtable("winequality-red.csv", 'PreserveVariableNames', 1);
data_white = readtable("winequality-white.csv", 'PreserveVariableNames', 1);
data_combined = [data_red;data_white];
data_combined = normalize(data_combined, 'range');

%% Specify predictor and response variables (Train test features x labels)
predictors = data_combined(:, 1:size(data_combined, 2)-1);
response = data_combined(:, size(data_combined, 2));

[x,y] = size(predictors);

predictorTrain = predictors(1:floor(size(data_combined,1))/5*4,:);
responseTrain = response(1:floor(size(data_combined,1))/5*4,:);

predictorTest = predictors(floor(size(predictors, 1)/5*4)+1:size(predictors, 1), :);
responseTest = response(floor(size(predictors, 1)/5*4)+1:size(response, 1), :);

%% (b) Performing inner cross-validation

% hyperparameters
kernelFunctions = ["linear", "rbf", "polynomial"];

boxConstraints = [0.1, 1, 5, 10, 20];

epsilonScale = [0.1, 0.3, 0.5, 0.7];
kernelScale = [0.1, 1, 5, 10, 20];
polynomialOrder = 2:4;

% Store Results
resultsNestedCV.linear = zeros(3, 1, length(epsilonScale), length(boxConstraints));
resultsNestedCV.rbf = zeros(3, length(kernelScale), length(epsilonScale), length(boxConstraints));    
resultsNestedCV.polynomial = zeros(3, length(polynomialOrder), length(epsilonScale), length(boxConstraints));

% take 10% of the original data for faster CV
% featuresNestedCV = predictorTrain(1:height(predictorTrain)/10, :);
% labelsNestedCV = responseTrain(1:height(responseTrain)/10, :);

featuresNestedCV = predictorTrain(1:floor(height(predictorTrain)/75), :);
labelsNestedCV = responseTrain(1:floor(height(responseTrain)/75), :);

% train SVM on nested cross-validation
numFoldsNestedCV = 5;
for f = kernelFunctions
    disp("Current kernel fn: " + f)
    count = 1;
    
    for c = boxConstraints
        fprintf("\n")
        disp("Current box constraint: "+c)
        
        for e = epsilonScale
            
            disp("Current epsilon: "+e)
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
                lowestOuterPoint = 1;
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
                    
                    lowestInnerPoint = 1;
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
                            modelRegression = fitrsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction',f, 'BoxConstraint',c, 'Epsilon', e, 'Standardize', true);
                        else
                            modelRegression = fitrsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction',f, 'BoxConstraint',c, funcArgName,val,'Epsilon', e, 'Standardize', true);
                        end

                        % evaluate SVM
                        [~, acc] = evaluateSVM(modelRegression, featuresFoldTestInner, labelsFoldTestInner);
                        point = acc;

                        if point < lowestInnerPoint
                            bestInnerModel = modelRegression;
                            lowestInnerPoint = point;
                        end
                    end

                    % evaluate best performing model
                    [~, acc] = evaluateSVM(bestInnerModel, featuresFoldTestOuter, labelsFoldTestOuter);
                    point = acc;
                    
                    if point < lowestOuterPoint
                        bestOuterModel = bestInnerModel;
                        lowestOuterPoint = point;
                    end
                end
            
            % store number and ratio of support vectors
            numSuppVec = size(bestOuterModel.SupportVectors, 1);
            suppVecRat = numSuppVec / height(featuresFoldTrainOuter) * 100; 

            if f == "linear"
                
                if numSuppVec == 0
                    lowestOuterPoint = 1;
                end
                
                resultsNestedCV.linear(count) = numSuppVec;
                resultsNestedCV.linear(count+1) = suppVecRat;
                resultsNestedCV.linear(count+2) = lowestOuterPoint;  
                
                
            elseif f == "rbf"  
                if numSuppVec == 0 % Prevent picking this one if no SV
                    lowestOuterPoint = 1;
                end
                
                resultsNestedCV.rbf(count) = numSuppVec;
                resultsNestedCV.rbf(count+1) = suppVecRat;
                resultsNestedCV.rbf(count+2) = lowestOuterPoint;   
                
            elseif f == "polynomial"
                if numSuppVec == 0
                    lowestOuterPoint = 1;
                end
                
                resultsNestedCV.polynomial(count) = numSuppVec;
                resultsNestedCV.polynomial(count+1) = suppVecRat;
                resultsNestedCV.polynomial(count+2) = lowestOuterPoint;  
            end
            count = count + 3; 
            end
        end
    end
end
%% Part (b) Pick hyperparams
bestHyperparamCombi = zeros(3,3);
countHyper = 1;

for f = kernelFunctions

    if f == "linear"
        resultMat = resultsNestedCV.linear;
        kfval = nan;
     
        [min1, bestEpIndex] = min(resultMat(3,:,:,:),[],3);  % Get best epsilon index
        [~, bestBoxCon] = min(min1); % Get the best box constraint index
        bestValArg = 1;
        
    elseif f == "rbf"
        resultMat = resultsNestedCV.rbf;
        kfval = kernelScale;
        
        [min1, bestEpIndex] = min(resultMat(3,:,:,:),[],3); % Get best epsilon index
        [min2, bestValArg] = min(min1); % get best argument index
        [~, bestBoxCon] = min(nonzeros(min2)'); % Get the best box constraint index
        
    elseif f == "polynomial"
        resultMat = resultsNestedCV.polynomial;
        kfval = polynomialOrder;
        
        [min1, bestEpIndex] = min(resultMat(3,:,:,:),[],3); % Get best epsilon index
        [min2, bestValArg] = min(min1); % get best argument index
        [~, bestBoxCon] = min(nonzeros(min2)'); % Get the best box constraint index
        
    end
    
    if f == "linear"
        bestHyperparamCombi(countHyper) = bestValArg;
    else
        bestHyperparamCombi(countHyper) = kfval(bestValArg(1)); 
    end
    bestHyperparamCombi(countHyper+1) = epsilonScale(bestEpIndex(1));
    bestHyperparamCombi(countHyper+2) = boxConstraints(bestBoxCon);
    
    countHyper = countHyper + 3;
     
    disp("----------------------------------------")
    disp("Best hyperparameter combination for " + f + " kernel function:")
    disp("  Box Constraint: " + boxConstraints(bestBoxCon))
    disp("  Kernel function argument value: " + kfval(bestValArg(1)))
    disp("  Best epsilon value found: " + epsilonScale(bestEpIndex(1)))
    disp("====================")
    disp("Best results from nested cross-validation:")
    disp("  Number of support vectors: " +  resultMat(1, bestValArg(bestBoxCon), bestEpIndex(1), bestBoxCon))    
    disp("  Support Vector Ratio: " + resultMat(2, bestValArg(bestBoxCon), bestEpIndex(1), bestBoxCon))
    disp("  Best RMSE: " +unique(min(resultMat(3, bestValArg, bestEpIndex, bestBoxCon))))
end

bestHyperparamCombi = bestHyperparamCombi'

%% Part(c) Perform 10-fold cross-validation for linear, gaussian rbf, and polynomial kernels
SVMPreds = [];
for f = 1:length(kernelFunctions)
    disp("Kernel: "+ kernelFunctions(f))
%     totalAcc = 0;
    totalRMSE = 0;
%     maxAcc = 0;
    minRMSE = 1;
    folds = 10;
    
    for fold = 1:folds
        disp("On fold: "+fold)
        % split dataset into training and testing datasets in each fold
        featuresFoldTest = predictorTrain((fold-1)*(floor(size(predictorTrain,1)/10))+1:fold*(floor(size(predictorTrain,1)/10)), :);
        featuresFoldTrain1 = predictorTrain(1:(fold-1)*(floor(size(predictorTrain,1)/10)), :);
        featuresFoldTrain2 = predictorTrain(fold*(floor(size(predictorTrain,1)/10))+1:size(predictorTrain,1), :);
        featuresFoldTrain = [featuresFoldTrain1; featuresFoldTrain2];
        
        labelsFoldTest = responseTrain((fold-1)*(floor(size(predictorTrain,1)/10))+1:fold*(floor(size(predictorTrain,1)/10)), :);
        labelsFoldTrain1 = responseTrain(1:(fold-1)*(floor(size(predictorTrain,1)/10)), :);
        labelsFoldTrain2 = responseTrain(fold*(floor(size(predictorTrain,1)/10))+1:size(responseTrain,1), :);
        labelsFoldTrain = [labelsFoldTrain1; labelsFoldTrain2];

        % train SVM
        if kernelFunctions(f) == "linear"
            modelRegression = fitrsvm(featuresFoldTrain, labelsFoldTrain, 'KernelFunction','linear', 'BoxConstraint',bestHyperparamCombi(f,3), 'Epsilon', bestHyperparamCombi(f,2), 'Standardize', true);
 
        elseif kernelFunctions(f) == "rbf"
            modelRegression = fitrsvm(featuresFoldTrain, labelsFoldTrain, 'KernelFunction','rbf', 'BoxConstraint',bestHyperparamCombi(f,3), 'KernelScale',bestHyperparamCombi(f,1), 'Epsilon', bestHyperparamCombi(f,2), 'Standardize', true);
            
        elseif kernelFunctions(f) == "polynomial"
            modelRegression = fitrsvm(featuresFoldTrain, labelsFoldTrain, 'KernelFunction','polynomial', 'BoxConstraint',bestHyperparamCombi(f,3), 'PolynomialOrder',bestHyperparamCombi(f,1), 'Epsilon', bestHyperparamCombi(f,2), 'Standardize', true);
        end

        % evaluate SVM
        [~, rmse] = evaluateSVM(modelRegression, featuresFoldTest, labelsFoldTest);

        % average and max accuracy
        if rmse < minRMSE
            minRMSE = rmse;
        end
        totalRMSE = totalRMSE + rmse;
        
    end
    
%     avgAcc = totalRMSE / folds * 100;
    avgRMSE = totalRMSE/folds;
    disp("----------------------------------------")
    disp("Result for " + kernelFunctions(f) + " kernel function in 10-fold cross-validation:")
    disp("  Min RMSE: " + minRMSE * 100)
    disp("  Average RMSE: " + avgRMSE)
    
    % train and evaluate best SVM model on whole dataset and retrieve its predictions
    if kernelFunctions(f) == "linear"
        modelRegression = fitrsvm(predictorTrain, responseTrain, 'KernelFunction','linear', 'BoxConstraint',bestHyperparamCombi(f,3), 'Epsilon', bestHyperparamCombi(f,2), 'Standardize', true);
        
    elseif kernelFunctions(f) == "rbf"
        modelRegression = fitrsvm(predictorTrain, responseTrain, 'KernelFunction','rbf', 'BoxConstraint',bestHyperparamCombi(f,3), 'KernelScale',bestHyperparamCombi(f,1), 'Epsilon', bestHyperparamCombi(f,2), 'Standardize', true);
    
    elseif kernelFunctions(f) == "polynomial"
        modelRegression = fitrsvm(predictorTrain, responseTrain, 'KernelFunction','polynomial', 'BoxConstraint',bestHyperparamCombi(f,3), 'PolynomialOrder',bestHyperparamCombi(f,1), 'Epsilon', bestHyperparamCombi(f,2), 'Standardize', true);
    end
    
    % evaluate best model for each kernel and store their results and predictions
    [preds,rmse] = evaluateSVM(modelRegression, predictorTest, responseTest);
    numSuppVec = size(modelRegression.SupportVectors, 1);
    disp("----------------------------------------")
    disp("Result from cross-validation training:")
    disp("  RMSE: " + rmse)
    disp("  Number of Support Vectors: " + numSuppVec)
    disp("  Support Vector Ratio: " + numSuppVec / height(predictors) * 100)
    
    SVMPreds = [SVMPreds preds];
    
end

%% Part (c) Ttest against existing work
%% evaluate SVM function
function [preds, acc] = evaluateSVM(model, features, labels)

preds = predict(model,features);

labs = table2array(labels);

acc = sqrt(immse(preds, labs));

end