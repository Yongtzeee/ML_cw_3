function SVM()
% load the dataset
data = readtable('online_shoppers_intention.csv');


%% Dataset manipulation for training
% take same number of true and false labels from original dataset
dataTrue = data(strcmp(data.Revenue, 'TRUE'), :);
numberLabelPerClass = size(dataTrue, 1);
dataFalse = data(randperm(size(data, 1), numberLabelPerClass), :);
data = [dataTrue ; dataFalse];

% shuffle dataset for consistency in data distribution
data = data(randperm(size(data, 1)), :);

% split dataset into features and labels
features = data(:, 1:size(data, 2)-1);
labels = data(:, size(data, 2));

[features, labels] = preProcessData(features, labels);


%% (a) Preliminary training to get it working
% train the SVM
modelClassification = fitcsvm(features, labels, 'KernelFunction','linear', 'BoxConstraint',1);
disp(modelClassification);
disp(modelClassification.Cost);
disp(modelClassification.ScoreTransform);
disp(modelClassification.SupportVectors);


%% (b) Performing inner cross-validation
boxConstraints = [0.1, 1, 100];
kernelFunctions = ["rbf", "polynomial"];
kernelScale = [0.1, 1, 100];
polynomialOrder = 2:4;

% train SVM on nestes cross-validation
numFoldsNestedCV = 5;
% take 10% of the original data for faster CV
featuresNestedCV = features(1:height(features)/10, :);
labelsNestedCV = labels(1:height(labels)/10, :);
for outerFold = 1:numFoldsNestedCV
    
    % split dataset into training and testing datasets in each fold
    featuresFoldTestOuter = featuresNestedCV((fold-1)*(floor(size(featuresNestedCV,1)/10))+1:fold*(floor(size(featuresNestedCV,1)/10)), :);
    featuresFoldTrain1 = featuresNestedCV(1:(fold-1)*(floor(size(featuresNestedCV,1)/10)), :);
    featuresFoldTrain2 = featuresNestedCV(fold*(floor(size(featuresNestedCV,1)/10))+1:size(featuresNestedCV,1), :);
    featuresFoldTrainOuter = [featuresFoldTrain1; featuresFoldTrain2];
    labelsFoldTestOuter = labelsNestedCV((fold-1)*(floor(size(featuresNestedCV,1)/10))+1:fold*(floor(size(featuresNestedCV,1)/10)), :);
    labelsFoldTrain1 = labelsNestedCV(1:(fold-1)*(floor(size(featuresNestedCV,1)/10)), :);
    labelsFoldTrain2 = labelsNestedCV(fold*(floor(size(featuresNestedCV,1)/10))+1:size(labelsNestedCV,1), :);
    labelsFoldTrainOuter = [labelsFoldTrain1; labelsFoldTrain2];
    
    for innerFold = 1:numFoldsNestedCV
        
        % split dataset into training and testing datasets in each fold
        featuresFoldTestInner = featuresFoldTrainOuter((fold-1)*(floor(size(featuresFoldTrainOuter,1)/10))+1:fold*(floor(size(featuresFoldTrainOuter,1)/10)), :);
        featuresFoldTrain1 = featuresFoldTrainOuter(1:(fold-1)*(floor(size(featuresFoldTrainOuter,1)/10)), :);
        featuresFoldTrain2 = featuresFoldTrainOuter(fold*(floor(size(featuresFoldTrainOuter,1)/10))+1:size(featuresFoldTrainOuter,1), :);
        featuresFoldTrainInner = [featuresFoldTrain1; featuresFoldTrain2];
        labelsFoldTestInner = labelsFoldTrainOuter((fold-1)*(floor(size(featuresFoldTrainOuter,1)/10))+1:fold*(floor(size(featuresFoldTrainOuter,1)/10)), :);
        labelsFoldTrain1 = labelsFoldTrainOuter(1:(fold-1)*(floor(size(featuresFoldTrainOuter,1)/10)), :);
        labelsFoldTrain2 = labelsFoldTrainOuter(fold*(floor(size(featuresFoldTrainOuter,1)/10))+1:size(labelsFoldTrainOuter,1), :);
        labelsFoldTrainInner = [labelsFoldTrain1; labelsFoldTrain2];
        
        for f = kernelFunctions
            for c = boxConstraints
                if f == "rbf"
                    for sigma = kernelScale
                        modelClassification = fitcsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction',f, 'BoxConstraint',c, 'KernelScale',sigma);
                        % store loss result
                        % Cost, ScoreTransform?, SupportVectors?
                        
                        % evaluate
                    end
                elseif f == "polynomial"
                    for q = polynomialOrder
                        modelClassification = fitcsvm(featuresFoldTrainInner, labelsFoldTrainInner, 'KernelFunction',f, 'BoxConstraint',c, 'PolynomialOrder',q);
                        % store loss result
                        
                        % evaluate
                    end
                end
            end
        end
    end
    
    % get best hyperparameter combination
    
        
    % evaluate best performing hyperparameter combination
    
end


% %% (c) Perform 10-fold cross-validation
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


end


% pre-process the dataset
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

end


