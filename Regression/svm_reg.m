%% Load data
data_red = readtable("winequality-red.csv", 'PreserveVariableNames', 1);
data_white = readtable("winequality-white.csv", 'PreserveVariableNames', 1);
data_combined = [data_red;data_white];
data_subset = data_combined(1:1000, :);

[x,y] = size(data_combined);

%% Specify predictor and response variables (Train test features x labels)
% predictorTrain = data_combined(1:floor(size(data_combined,1))/5*4,1:y-1);
% responseTrain = data_combined(1:floor(size(data_combined,1))/5*4,y);
% predictorTest = data_combined(floor(size(data_combined,1)/5*4)+1:x ,1:y-1);
% responseTest = data_combined(floor(size(data_combined,1)/5*4)+1:x ,y);

predictorTrain = data_subset(1:750,1:y-1);
responseTrain = data_subset(1:750,y);

predictorTest = data_subset(750:size(data_subset,1),1:y-1);
responseTest = data_subset(750:size(data_subset,1),y);

%% Linear kernel training
Mdl = fitrsvm(predictorTrain, responseTrain,'KernelFunction', 'linear' ,'Epsilon', 0.5, 'Standardize', true);
acc = predict(Mdl, predictorTest);

disp("Convergence: " + Mdl.ConvergenceInfo.Converged)
disp(" # of support vectors: " + size(Mdl.SupportVectors,1))

%% TO WRITE CV FOR HYPERPARAMS FINDINGS

% Gaussian kernel


% RBF kernel 


% Polynomial kernel
