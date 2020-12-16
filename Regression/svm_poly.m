%% Load data
data_red = readtable("winequality-red.csv", 'PreserveVariableNames', 1);
data_white = readtable("winequality-white.csv", 'PreserveVariableNames', 1);
data_combined = [data_red;data_white];
data_subset = data_combined(1:2000, :);

[x,y] = size(data_combined);

%% Specify predictor and response variables (Train test features x labels)
predictorTrain = data_combined(1:floor(size(data_combined,1))/5*4,1:y-1);
responseTrain = data_combined(1:floor(size(data_combined,1))/5*4,y);

predictorTest = data_combined(floor(size(data_combined,1)/5*4)+1:x ,1:y-1);
responseTest = data_combined(floor(size(data_combined,1)/5*4)+1:x ,y);

%% Part (b) Train each model -> Linear, RBF, Polynomial

% Reduce number of datapoints for CV
predictorCV = predictorTrain(1:height(predictorTrain)/10, :);
responseCV = responseTrain(1:height(responseTrain)/10, :);

cvFolds = 5;

% Hyperparameters declaration for regression
boxConstraints = [0.1, 1, 5, 10, 20];
epsilonScale = [0.3, 0.5, 0.7, 0.9];

%% Polynomial Kernel

% Store results
resultsNestedCV.polynomial = zeros(3, 1, length(boxConstraints));

polynomialOrder = 2:2;
count = 1;

for c = boxConstraints
    bestBestBestRMSE = 1;
    bestBestBestModel = 0;
    bestEpsilon = 0;
    
    for ep = epsilonScale
        fprintf("\n")
        disp("Current ep: "+ep)
        
        bestPolynomialOrder = 0;
        bestBestRMSE = 1;
        bestBestModel = 0;
        
        for k = polynomialOrder
            disp("Polynomial Order: "+k)
            bestRMSE = 1;
%             bestKernelModel = 0;
            
            for outerFold = 1:cvFolds
                % split dataset into training and testing datasets in each fold
                featuresFoldTestOuter = predictorCV((outerFold-1)*(floor(size(predictorCV,1)/10))+1:outerFold*(floor(size(predictorCV,1)/10)), :);
                featuresFoldTrain1 = predictorCV(1:(outerFold-1)*(floor(size(predictorCV,1)/10)), :);
                featuresFoldTrain2 = predictorCV(outerFold*(floor(size(predictorCV,1)/10))+1:size(predictorCV,1), :);
                featuresFoldTrainOuter = [featuresFoldTrain1; featuresFoldTrain2];

                labelsFoldTestOuter = responseCV((outerFold-1)*(floor(size(predictorCV,1)/10))+1:outerFold*(floor(size(predictorCV,1)/10)), :);
                labelsFoldTrain1 = responseCV(1:(outerFold-1)*(floor(size(predictorCV,1)/10)), :);
                labelsFoldTrain2 = responseCV(outerFold*(floor(size(predictorCV,1)/10))+1:size(responseCV,1), :);
                labelsFoldTrainOuter = [labelsFoldTrain1; labelsFoldTrain2];
                
                lowestInnerFoldRMSE = 1;
                bestInnerFoldModel = 0;
                
                for innerFold = 1:cvFolds
                    disp("Inner fold: "+innerFold)
                    % split dataset into training and testing datasets in each fold
                    featuresFoldTestInner = featuresFoldTrainOuter((innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10))+1:innerFold*(floor(size(featuresFoldTrainOuter,1)/10)), :);
                    featuresFoldTrain1 = featuresFoldTrainOuter(1:(innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10)), :);
                    featuresFoldTrain2 = featuresFoldTrainOuter(innerFold*(floor(size(featuresFoldTrainOuter,1)/10))+1:size(featuresFoldTrainOuter,1), :);
                    featuresFoldTrainInner = [featuresFoldTrain1; featuresFoldTrain2];

                    labelsFoldTestInner = labelsFoldTrainOuter((innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10))+1:innerFold*(floor(size(featuresFoldTrainOuter,1)/10)), :);
                    labelsFoldTrain1 = labelsFoldTrainOuter(1:(innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10)), :);
                    labelsFoldTrain2 = labelsFoldTrainOuter(innerFold*(floor(size(featuresFoldTrainOuter,1)/10))+1:size(labelsFoldTrainOuter,1), :);
                    labelsFoldTrainInner = [labelsFoldTrain1; labelsFoldTrain2];
                
                    % Train model
                    modelRegression = fitrsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction', 'polynomial', 'PolynomialOrder', k, 'BoxConstraint',c, 'Epsilon',ep ,'Standardize', 1);
                    
                    % Evaluate the model
                    predicted_labels = predict(modelRegression, featuresFoldTestInner);
                    mse = immse(table2array(labelsFoldTestInner), predicted_labels);
                    rmse = sqrt(mse);
                    
                    disp("RMSE calculated: "+rmse)
                    disp("Lowest Inner fold RMSE: "+lowestInnerFoldRMSE)
                    
                    if rmse < lowestInnerFoldRMSE
                        lowestInnerFoldRMSE = rmse;
                        bestInnerFoldModel = modelRegression;
                    end
                end
                
                % Evaluate the best inner fold model
                predicted_labels = predict(bestInnerFoldModel, featuresFoldTestOuter);
                mse = immse(table2array(labelsFoldTestOuter), predicted_labels);
                rmse = sqrt(mse);
                
                if rmse < bestRMSE
                    bestRMSE = rmse;
                    bestModel = bestInnerFoldModel;
                end  
            end
            disp("BestRMSE: "+bestRMSE)
            disp("BestBestRMSE: "+bestBestRMSE)
           
            if bestRMSE < bestBestRMSE
                bestKernelScale = k;
                bestBestRMSE = bestRMSE;
                bestBestModel = bestModel;
                disp("New best kernel scale: "+bestKernelScale)
            end
            fprintf("\n")
        end  
    end
    if bestBestRMSE < bestBestBestRMSE
        bestBestBestRMSE = bestBestRMSE;
        bestEpsilon = ep;
        bestBestBestModel = bestBestModel;
    end
    disp("Triple best RMSE: "+bestBestBestRMSE)
    
    suppVecNum = size(bestBestBestModel.SupportVectors,1);
    suppVecRat = suppVecNum/height(featuresFoldTrainOuter) * 100;
    
    resultsNestedCV.polynomial(count) = suppVecNum;
    resultsNestedCV.polynomial(count+1) = suppVecRat;
    resultsNestedCV.polynomial(count+2) = bestBestBestRMSE;
    
    count = count+3;  
end

bestHyperparamCombi_poly = zeros(2,1); % confirm size
countHyper = 1;
bc = boxConstraints;

% get highest point for each model
[maxPointsDim2, idx2] = max(resultsNestedCV.polynomial(3,:,:), [], 1);
[~, idx3] = max(maxPointsDim2, [], 3);

bestHyperparamCombi_poly(countHyper) = polynomialOrder(idx2(idx3));
bestHyperparamCombi_poly(countHyper+1) = bc(idx3);

countHyper = countHyper + 2;

disp("----------------------------------------")
disp("Best hyperparameter combination for polynomial kernel function:")
disp("  Box Constraint: " + bc(idx3))
disp("  Kernel Function argument value: " + polynomialOrder(idx2(idx3)))
disp("====================")
disp("Best results from nested cross-validation:")
disp("  Number of support vectors: " + resultsNestedCV.polynomial(1,idx2(idx3),idx3))
disp("  Support vector ratio: " + resultsNestedCV.polynomial(2,idx2(idx3),idx3))
disp("  RMSE: " + resultsNestedCV.polynomial(3,idx2(idx3),idx3))

bestHyperparamCombi_poly = bestHyperparamCombi_poly';

%% perform 10-fold cross validation for polynomial kernel
SVMPreds = [];

folds = 10;
bestRMSE = 0;
totalRMSE = 0; % for average RMSE of all 10 folds

for fold = 1:folds
    
    % split dataset into training and testing datasets in each fold
    featuresFoldTest = predictorCV((fold-1)*(floor(size(predictorCV,1)/10))+1:fold*(floor(size(predictorCV,1)/10)), :);
    featuresFoldTrain1 = predictorCV(1:(fold-1)*(floor(size(predictorCV,1)/10)), :);
    featuresFoldTrain2 = predictorCV(fold*(floor(size(predictorCV,1)/10))+1:size(predictorCV,1), :);
    featuresFoldTrain = [featuresFoldTrain1; featuresFoldTrain2];
    
    labelsFoldTest = responseCV((fold-1)*(floor(size(predictorCV,1)/10))+1:fold*(floor(size(predictorCV,1)/10)), :);
    labelsFoldTrain1 = responseCV(1:(fold-1)*(floor(size(predictorCV,1)/10)), :);
    labelsFoldTrain2 = responseCV(fold*(floor(size(predictorCV,1)/10))+1:size(responseCV,1), :);
    labelsFoldTrain = [labelsFoldTrain1; labelsFoldTrain2];
    
    % train SVM
    modelRegression = fitrsvm(featuresFoldTrain, labelsFoldTrain, 'KernelFunction','polynomial', 'BoxConstraint',bestHyperparamCombi_poly('polynomial',2), 'PolynomialOrder',bestHyperparamCombi_poly('polynomial',1));
    
    % evaluate SVM
    predicted_labels = predict(modelRegression, featuresFoldTest);
    mse = immse(table2array(labelsFoldTest), predicted_labels);
    rmse = sqrt(mse);
    
    % average and best RMSE
    if rmse < bestRMSE
        bestRMSE = rmse;
    end
    totalRMSE = totalRMSE + rmse;
end

% results for 10 fold CV
avgRMSE = totalRMSE / folds;
disp("----------------------------------------")
disp("Result for polynomial kernel function in 10-fold cross-validation:")
disp("  Max RMSE: " + bestRMSE)
disp("  Average RMSE: " + avgRMSE)

% train and evaluate best SVM model on whole dataset and retrieve its predictions
modelRegression = fitrsvm(predictorTrain, responseTrain, 'KernelFunction','polynomial', 'BoxConstraint',bestHyperparamCombi_poly(polynomial,2), 'PolynomialOrder',bestHyperparamCombi_poly(polynomial,1));

% evaluate best model for polynomial kernel and store their results and predictions
predicted_labels = predict(modelRegression, predictorTest);
mse = immse(table2array(responseTest), predicted_labels);
% numSuppVec = size(modelClassification.SupportVectors, 1);
suppVecNum = size(modelRegression.SupportVectors,1);
disp("----------------------------------------")
disp("Result from cross-validation training:")
disp("  RMSE: " + bestRMSE)
disp("  Number of Support Vectors: " + suppVecNum)
% disp("  Support Vector Ratio: " + suppVecNum/height(features) * 100)