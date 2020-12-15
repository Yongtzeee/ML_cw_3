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

%% Linear kernel training (commented out to save time for now)
% Mdl = fitrsvm(predictorTrain, responseTrain,'KernelFunction', 'linear' ,'Epsilon', 0.5, 'Standardize', true);
% 
% % Test on test set 
% acc = predict(Mdl, predictorTest);
% 
% % Calculate MSE of the predicted set against GT
% mse = immse(table2array(responseTest), acc);
% rmse = sqrt(mse);
% 
% disp("Convergence: " + Mdl.ConvergenceInfo.Converged)
% disp(" # of support vectors: " + size(Mdl.SupportVectors,1))
% disp("RMSE: " + rmse)

%% Part (b) Train each model -> Linear, RBF, Polynomial

% Reduce number of datapoints for CV
predictorCV = predictorTrain(1:height(predictorTrain)/10, :);
responseCV = responseTrain(1:height(responseTrain)/10, :);

cvFolds = 5;

% Hyperparameters declaration for regression
boxConstraints = [0.1, 1, 5, 10, 20];
epsilonScale = [0.3, 0.5, 0.7, 0.9];

% Store results
% resultsNestedCV.linear = zeros(3, 1, length(boxConstraints));

%% RBF Kernel
kernelScale = [0.1, 1, 5, 10, 20];

resultsNestedCV.rbf = zeros(3, 1, length(boxConstraints));    
count = 1;
for c = boxConstraints
    bestBestBestRMSE = 1;
    bestBestBestModel = 0;
    bestEpsilon = 0;
    
    for ep = epsilonScale
        fprintf("\n")
        disp("Current ep: "+ep)
        
        bestKernelScale = 0;
        bestBestRMSE = 1;
        bestBestModel = 0;
        
        for k = kernelScale
            bestRMSE = 1;
            bestKernelModel = 0;
            
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
                    modelRegression = fitrsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction', 'rbf', 'BoxConstraint',c, 'Epsilon',ep);
                    
                    % Evaluate the model
                    predicted_labels = predict(modelRegression, featuresFoldTestInner);
                    mse = immse(table2array(labelsFoldTestInner), predicted_labels);
                    rmse = sqrt(mse);
                    
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
    
    resultsNestedCV.rbf(count) = suppVecNum;
    resultsNestedCV.rbf(count+1) = suppVecRat;
    resultsNestedCV.rbf(count+2) = bestBestBestRMSE;
    
    count = count+3;
    
end
%% Polynomial Kernel
% polynomialOrder = 2:4;
% resultsNestedCV.polynomial = zeros(5, length(polynomialOrder), length(boxConstraints));
% 
% for c = boxConstraints
%     for ep = epsilonScale
%         fprintf("\n")
%         disp("Current ep: "+ep)
%         
%         bestPolynomialOrder = 0;
%         bestBestRMSE = 1;
%         
%         for k = polynomialOrder
%             bestRMSE = 1;
%             
%             for outerFold = 1:cvFolds
%                 % split dataset into training and testing datasets in each fold
%                 featuresFoldTestOuter = predictorCV((outerFold-1)*(floor(size(predictorCV,1)/10))+1:outerFold*(floor(size(predictorCV,1)/10)), :);
%                 featuresFoldTrain1 = predictorCV(1:(outerFold-1)*(floor(size(predictorCV,1)/10)), :);
%                 featuresFoldTrain2 = predictorCV(outerFold*(floor(size(predictorCV,1)/10))+1:size(predictorCV,1), :);
%                 featuresFoldTrainOuter = [featuresFoldTrain1; featuresFoldTrain2];
% 
%                 labelsFoldTestOuter = responseCV((outerFold-1)*(floor(size(predictorCV,1)/10))+1:outerFold*(floor(size(predictorCV,1)/10)), :);
%                 labelsFoldTrain1 = responseCV(1:(outerFold-1)*(floor(size(predictorCV,1)/10)), :);
%                 labelsFoldTrain2 = responseCV(outerFold*(floor(size(predictorCV,1)/10))+1:size(responseCV,1), :);
%                 labelsFoldTrainOuter = [labelsFoldTrain1; labelsFoldTrain2];
%                 
%                 lowestInnerFoldRMSE = 1;
%                 bestInnerFoldModel = 0;
%                 
%                 for innerFold = 1:cvFolds
%                     % split dataset into training and testing datasets in each fold
%                     featuresFoldTestInner = featuresFoldTrainOuter((innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10))+1:innerFold*(floor(size(featuresFoldTrainOuter,1)/10)), :);
%                     featuresFoldTrain1 = featuresFoldTrainOuter(1:(innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10)), :);
%                     featuresFoldTrain2 = featuresFoldTrainOuter(innerFold*(floor(size(featuresFoldTrainOuter,1)/10))+1:size(featuresFoldTrainOuter,1), :);
%                     featuresFoldTrainInner = [featuresFoldTrain1; featuresFoldTrain2];
% 
%                     labelsFoldTestInner = labelsFoldTrainOuter((innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10))+1:innerFold*(floor(size(featuresFoldTrainOuter,1)/10)), :);
%                     labelsFoldTrain1 = labelsFoldTrainOuter(1:(innerFold-1)*(floor(size(featuresFoldTrainOuter,1)/10)), :);
%                     labelsFoldTrain2 = labelsFoldTrainOuter(innerFold*(floor(size(featuresFoldTrainOuter,1)/10))+1:size(labelsFoldTrainOuter,1), :);
%                     labelsFoldTrainInner = [labelsFoldTrain1; labelsFoldTrain2];
%                 
%                     % Train model
%                     modelRegression = fitrsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction', 'rbf', 'BoxConstraint',c, 'Epsilon',ep);
%                     
%                     % Evaluate the model
%                     predicted_labels = predict(modelRegression, featuresFoldTestInner);
%                     mse = immse(table2array(labelsFoldTestInner), predicted_labels);
%                     rmse = sqrt(mse);
%                     
%                     if rmse < lowestInnerFoldRMSE
%                         lowestInnerFoldRMSE = rmse;
%                         bestInnerFoldModel = modelRegression;
%                     
%                     end
%                 end
%                 
%                 % Evaluate the best inner fold model
%                 predicted_labels = predict(bestInnerFoldModel, featuresFoldTestOuter);
%                 mse = immse(table2array(labelsFoldTestOuter), predicted_labels);
%                 rmse = sqrt(mse);
%                 
%                 if rmse < bestRMSE
%                     bestRMSE = rmse;
%                 end  
%             end
%             
%             disp("BestRMSE: "+bestRMSE)
%             disp("BestBestRMSE: "+bestBestRMSE)
%            
%             if bestRMSE < bestBestRMSE
%                 bestPolynomialOrder = k;
%                 bestBestRMSE = bestRMSE;
%                 disp("New best kernel scale: "+bestPolynomialOrder)
%             end
%             fprintf("\n")
%         end 
%     end
% end
