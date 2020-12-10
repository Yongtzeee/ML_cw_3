function modify_csv()
% This function modifies the classification dataset such that all three ML 
% models (ANN, Decision Tree, and SVM) train and predict from the same 
% dataset. This is to ensure the statistical test can be conducted.

% load the dataset
data = readtable('online_shoppers_intention.csv');

% take same number of true and false labels from original dataset
dataTrue = data(strcmp(data.Revenue, 'TRUE'), :);
numberLabelPerClass = size(dataTrue, 1);
dataFalse = data(randperm(size(data, 1), numberLabelPerClass), :);
data = [dataTrue ; dataFalse];

% shuffle dataset for consistency in data distribution
data = data(randperm(size(data, 1)), :);

% write the modified dataset into storage
writetable(data, "online_shoppers_intention_mod.csv");

end