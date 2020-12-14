%% Load data
data_red = readtable("winequality-red.csv", 'PreserveVariableNames', 1);
data_white = readtable("winequality-white.csv", 'PreserveVariableNames', 1);
data_combined = [data_red;data_white];
data_subset = data_combined(1:2000, :);

[x,y] = size(data_combined);

%% Specify predictor and response variables (Train test features x labels)
% predictorTrain = data_combined(1:floor(size(data_combined,1))/5*4,1:y-1);
% responseTrain = data_combined(1:floor(size(data_combined,1))/5*4,y);
% predictorTest = data_combined(floor(size(data_combined,1)/5*4)+1:x ,1:y-1);
% responseTest = data_combined(floor(size(data_combined,1)/5*4)+1:x ,y);

predictorTrain = data_subset(1:1500,1:y-1);
responseTrain = data_subset(1:1500,y);

predictorTest = data_subset(1500:size(data_subset,1),1:y-1);
responseTest = data_subset(1500:size(data_subset,1),y);

%% Linear kernel training
Mdl = fitrsvm(predictorTrain, responseTrain,'KernelFunction', 'linear' ,'Epsilon', 0.5, 'Standardize', true);

% Test on test set 
acc = predict(Mdl, predictorTest);

% Calculate MSE of the predicted set against GT
mse = immse(table2array(responseTest), acc);
rmse = sqrt(mse);

disp("Convergence: " + Mdl.ConvergenceInfo.Converged)
disp(" # of support vectors: " + size(Mdl.SupportVectors,1))
disp("RMSE: " + rmse)

%% TO WRITE CV FOR HYPERPARAMS FINDINGS

% Hyperparameters declaration
boxConstraints = [0.1, 1, 5, 10, 20];
kernelFunctions = ["linear", "rbf", "polynomial"];
kernelScale = [0.1, 1, 5, 10, 20];
polynomialOrder = 2:4;
epsilonScale = [0.3, 0.4, 0.5, 0.6, 0.7];

% Store results
resultsNestedCV.linear = zeros(4, 1, length(boxConstraints));

% number of SV, ratio of SV, accuracy, points for both
resultsNestedCV.rbf = zeros(4, length(kernelScale), length(boxConstraints));    
resultsNestedCV.polynomial = zeros(4, length(polynomialOrder), length(boxConstraints));

% Reduce number of datapoints for CV
predictorCV = predictorTrain(1:height(predictorTrain)/10, :);
responseCV = responseTrain(1:height(responseTrain)/10, :);

cvFolds = 5;

for f = kernelFunctions
    count = 1;
    
    for c = boxConstraints
        if f == "linear"        
            argsVals = 1:1 ;
            
        elseif f == "rbf"
            argsName = "KernelScale";
            argsVals = kernelScale;
            
        elseif f == "polynomial"
            argsName = "PolynomialOrder";
            argsVals = polynomialOrder;       
        end
        
        for val = argsVals
            highestOuterPoint = 0;
            bestOuterModel = 0;
            bestAcc = 0;
            
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
                
                highestInnerPoint = 0;
                bestInnerModel = 0;
                
                for innerFold = 1:cvFolds
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
                        modelRegression = fitrsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction',f, 'BoxConstraint',c);
                    else
                        modelRegression = fitrsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction',f, 'BoxConstraint',c, funcArgName,val);
                    end
                    
                    % Evaluate SVM on fold test set 
                    acc = predict(modelRegression, featuresFoldTestInner);
                    
                    % Calculate MSE of the predicted set against GT
                    mse = immse(table2array(labelsFoldTestInner), acc);
                    rmse = sqrt(mse);
                    
                    disp("Inner fold: " + innerFold);
                    disp("Outer fold: " + outerFold);
                    disp("Kernel Function: " + f);
                    disp("Box Constraints: " + c);
                    disp("Num of support vectors: " + size(modelRegression.SupportVectors,1))
                end  
            end    
        end
    end
end


% Gaussian kernel


% RBF kernel 


% Polynomial kernel
