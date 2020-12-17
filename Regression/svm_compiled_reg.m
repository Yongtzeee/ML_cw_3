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

bestHyperparamCombi = bestHyperparamCombi';

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
%% compare results between ANN, DT and SVM
% get predictions for Decision Tree and ANN
ANNPreds = [0.5261600613594055, 0.40120288729667664, 0.3200933039188385, 0.4675559401512146, 0.5457885265350342, 0.5841102600097656, 0.4375443458557129, 0.4961181879043579, 0.38872432708740234, 0.4695396423339844, 0.5741469264030457, 0.5167111754417419, 0.27704012393951416, 0.5667243003845215, 0.3858437240123749, 0.6828852295875549, 0.30757972598075867, 0.48000332713127136, 0.3830947279930115, 0.490254670381546, 0.41527536511421204, 0.4462665021419525, 0.43639451265335083, 0.4243086278438568, 0.6112545728683472, 0.5810712575912476, 0.5667515397071838, 0.35131537914276123, 0.4555865228176117, 0.5782383680343628, 0.6556347608566284, 0.47604990005493164, 0.3704416751861572, 0.5990619659423828, 0.5042205452919006, 0.5110657215118408, 0.3728005886077881, 0.41563841700553894, 0.6633560061454773, 0.40444785356521606, 0.49804750084877014, 0.2919250726699829, 0.6064841747283936, 0.5189568400382996, 0.5879796743392944, 0.33400803804397583, 0.5784795880317688, 0.4699456989765167, 0.3990326523780823, 0.5802720785140991, 0.5323567390441895, 0.43489426374435425, 0.4153295159339905, 0.5671543478965759, 0.5103511810302734, 0.47525718808174133, 0.49910813570022583, 0.6585047245025635, 0.38558995723724365, 0.4590027332305908, 0.45615172386169434, 0.4816814959049225, 0.42923814058303833, 0.5905899405479431, 0.6569741368293762, 0.467374324798584, 0.3173314332962036, 0.26797762513160706, 0.49128279089927673, 0.4404599368572235, 0.45615172386169434, 0.35707545280456543, 0.40007078647613525, 0.5781934261322021, 0.5343866348266602, 0.3545793890953064, 0.7062817811965942, 0.3309418261051178, 0.557550847530365, 0.4391026496887207, 0.4432917833328247, 0.4189817011356354, 0.45579060912132263, 0.5302191972732544, 0.565220296382904, 0.5796089172363281, 0.28190553188323975, 0.5694962739944458, 0.5872516632080078, 0.41907402873039246, 0.3224482238292694, 0.3366627097129822, 0.47265109419822693, 0.6510543823242188, 0.4823102056980133, 0.4436424672603607, 0.39302971959114075, 0.6203659176826477, 0.4657417833805084, 0.3412904739379883, 0.4125405251979828, 0.4844859838485718, 0.42966869473457336, 0.5863077044487, 0.6387800574302673, 0.27665331959724426, 0.35529613494873047, 0.6690658926963806, 0.6636030673980713, 0.3658561706542969, 0.5152493119239807, 0.44167810678482056, 0.4825890064239502, 0.3162442445755005, 0.41813623905181885, 0.44681453704833984, 0.33333396911621094, 0.48142310976982117, 0.3124886453151703, 0.6006036400794983, 0.5586096048355103, 0.5550317168235779, 0.4388144612312317, 0.38748592138290405, 0.34572386741638184, 0.5906597375869751, 0.28635960817337036, 0.522780179977417, 0.3134472370147705, 0.3495737314224243, 0.3711145520210266, 0.39227554202079773, 0.46702128648757935, 0.45122164487838745, 0.396692156791687, 0.5818577408790588, 0.29764920473098755, 0.5562347173690796, 0.4504532217979431, 0.4967426657676697, 0.295825719833374, 0.5836001038551331, 0.4422611594200134, 0.49453943967819214, 0.35890763998031616, 0.446846067905426, 0.4727497696876526, 0.5078211426734924, 0.4105774462223053, 0.6294469833374023, 0.4370745122432709, 0.6073483228683472, 0.5000147819519043, 0.5012593865394592, 0.5253916382789612, 0.5343866348266602, 0.3948308229446411, 0.36335888504981995, 0.36411815881729126, 0.6574880480766296, 0.40542760491371155, 0.6968685388565063, 0.5495742559432983, 0.2147180438041687, 0.4919082224369049, 0.445491760969162, 0.5941331386566162, 0.6048460006713867, 0.6055949926376343, 0.36707475781440735, 0.46840307116508484, 0.41415756940841675, 0.4736393988132477, 0.46059638261795044, 0.5837121605873108, 0.24091991782188416, 0.2697420120239258, 0.63605797290802, 0.547271192073822, 0.4300054609775543, 0.36302801966667175, 0.40655073523521423, 0.5317848920822144, 0.657497227191925, 0.43590736389160156, 0.312163770198822, 0.4937589168548584, 0.5434523820877075, 0.4672573506832123, 0.41648048162460327, 0.5259625911712646, 0.46723026037216187, 0.5361824631690979, 0.6590075492858887, 0.7074252367019653, 0.6029637455940247, 0.2803649306297302, 0.5206534266471863, 0.41804108023643494, 0.6219921112060547, 0.6436546444892883, 0.5431473851203918, 0.4403708875179291, 0.4912389814853668, 0.6259562969207764, 0.5145731568336487, 0.5764763355255127, 0.6595240235328674, 0.6646933555603027, 0.4408402144908905, 0.6332296133041382, 0.4032585024833679, 0.4985124170780182, 0.6569741368293762, 0.3115655779838562, 0.535487949848175, 0.6803895235061646, 0.6966672539710999, 0.4501943588256836, 0.2412150800228119, 0.6458656191825867, 0.5561705827713013, 0.5308482646942139, 0.6816986799240112, 0.6333103775978088, 0.6086300611495972, 0.4196077287197113, 0.42295220494270325, 0.45910176634788513, 0.5352834463119507, 0.48951131105422974, 0.6429340839385986, 0.5899490118026733, 0.5499932765960693, 0.5390530228614807, 0.6123579144477844, 0.5423130989074707, 0.4700682461261749, 0.576384961605072, 0.5927092432975769, 0.684110701084137, 0.36960849165916443, 0.5179595947265625, 0.3573492169380188, 0.4598073661327362, 0.32841235399246216, 0.4651889204978943, 0.28143543004989624, 0.4325372576713562, 0.5522688031196594, 0.33445703983306885, 0.6849350929260254, 0.40715816617012024, 0.3550620675086975, 0.5847194194793701, 0.6521621346473694, 0.5909907221794128, 0.490434855222702, 0.4450107216835022, 0.40869614481925964, 0.32736730575561523, 0.5262137055397034, 0.506047785282135, 0.5072311162948608, 0.5442919731140137, 0.352660208940506, 0.352486252784729, 0.6435980200767517, 0.6595592498779297, 0.5666725039482117, 0.4482671618461609, 0.3098146319389343, 0.34663885831832886, 0.48421692848205566, 0.5107438564300537, 0.6239017844200134, 0.4675166606903076, 0.30633023381233215, 0.4629102647304535, 0.3753436505794525, 0.45303499698638916, 0.5439993143081665, 0.49354469776153564, 0.5637664198875427, 0.38771504163742065, 0.48288846015930176, 0.6085163354873657, 0.5495058298110962, 0.3288329243659973, 0.6072952747344971, 0.3351261019706726, 0.3794339895248413, 0.45511844754219055, 0.3408341407775879, 0.5070716738700867, 0.5311414003372192, 0.32517439126968384, 0.47630074620246887, 0.3419736921787262, 0.44438016414642334, 0.4300486147403717, 0.3299901783466339, 0.6476003527641296, 0.4144623279571533, 0.4911620020866394, 0.5097146034240723, 0.45667341351509094, 0.6384516954421997, 0.196810781955719, 0.43670955300331116, 0.5340166687965393, 0.2948414087295532, 0.5216181874275208, 0.5149869322776794, 0.6858167052268982, 0.45108652114868164, 0.655616044998169, 0.5284370183944702, 0.3792092502117157, 0.5954623222351074, 0.43643540143966675, 0.45876920223236084, 0.4551507532596588, 0.5496789216995239, 0.2614082098007202, 0.4457470774650574, 0.6785227060317993, 0.24609321355819702, 0.46509766578674316, 0.42037469148635864, 0.3818025588989258, 0.4729098379611969, 0.5891110301017761, 0.2989501655101776, 0.48926377296447754, 0.5323225259780884, 0.6612129211425781, 0.4511417746543884, 0.4777868986129761, 0.5079678297042847, 0.3921515941619873, 0.4147108793258667, 0.5068932175636292, 0.44644880294799805, 0.4965340197086334, 0.6403655409812927, 0.44143182039260864, 0.42437034845352173, 0.4293951392173767, 0.5282363295555115, 0.4758549630641937, 0.40124237537384033, 0.5388972759246826, 0.42747920751571655, 0.5010078549385071, 0.3434697985649109, 0.4512011706829071, 0.36226609349250793, 0.43772822618484497, 0.4612765312194824, 0.24772527813911438, 0.4647120535373688, 0.4333422780036926, 0.6988012790679932, 0.6825095415115356, 0.3383372724056244, 0.41333121061325073, 0.617625892162323, 0.5837121605873108, 0.4495394825935364, 0.4213801622390747, 0.4215412139892578, 0.5288392305374146, 0.4983818233013153, 0.5948993563652039, 0.6144049167633057, 0.5085998177528381, 0.5585582852363586, 0.4353618621826172, 0.5347495675086975, 0.44205242395401, 0.4068782925605774, 0.45961445569992065, 0.38867437839508057, 0.31357520818710327, 0.470465749502182, 0.40904295444488525, 0.6333916187286377, 0.392878919839859, 0.5623993277549744, 0.46015653014183044, 0.5397814512252808, 0.5379400253295898, 0.29653918743133545, 0.3188077211380005, 0.5954078435897827, 0.45858800411224365, 0.44583815336227417, 0.6767158508300781, 0.2724318504333496, 0.43320387601852417, 0.546144425868988, 0.36005449295043945, 0.4426424205303192, 0.39291220903396606, 0.4160539209842682, 0.6011699438095093, 0.4030546545982361, 0.6029053926467896, 0.5042227506637573, 0.6474472880363464, 0.43840569257736206, 0.6392924785614014, 0.6239023208618164, 0.39112967252731323, 0.4510381817817688, 0.4856988489627838, 0.43302714824676514, 0.6341468691825867, 0.46312081813812256, 0.43643540143966675, 0.2890034317970276, 0.472648024559021, 0.31849557161331177, 0.4071807861328125, 0.4224320650100708, 0.47798556089401245, 0.46223148703575134, 0.6693235039710999, 0.44681453704833984, 0.43378013372421265, 0.574294924736023, 0.6252449750900269, 0.45651888847351074, 0.3977001905441284, 0.5451647043228149, 0.26713278889656067, 0.5238355398178101, 0.4105774462223053, 0.4557151198387146, 0.4268055260181427, 0.43314340710639954, 0.28747034072875977, 0.3756374716758728, 0.5226369500160217, 0.4652441740036011, 0.5559771656990051, 0.38373640179634094, 0.6577700972557068, 0.39868029952049255, 0.42486101388931274, 0.6211584210395813, 0.3911440968513489, 0.40342849493026733, 0.32308924198150635, 0.6070375442504883, 0.42108336091041565, 0.6575993299484253, 0.4349587857723236, 0.4380990266799927, 0.6539101600646973, 0.5174342393875122, 0.6468141078948975, 0.6113190650939941, 0.5563896298408508, 0.5584511160850525, 0.4376031756401062, 0.6329618692398071, 0.4946858584880829, 0.6245887279510498, 0.5671502351760864, 0.617625892162323, 0.33322906494140625, 0.42776525020599365, 0.6018519997596741, 0.4071465730667114, 0.5178585648536682, 0.5009909272193909, 0.42281049489974976, 0.5626515746116638, 0.6270471811294556, 0.5962476134300232, 0.4905547499656677, 0.3997385799884796, 0.40819448232650757, 0.4418092370033264, 0.45971110463142395, 0.49804750084877014, 0.4075901210308075, 0.5963572263717651, 0.5929567813873291, 0.3216626048088074, 0.612797200679779, 0.4657800495624542, 0.6094541549682617, 0.5223404169082642, 0.5479727983474731, 0.4633841812610626, 0.521035373210907, 0.38065317273139954, 0.4877229332923889, 0.5028511881828308, 0.5867511034011841, 0.48324453830718994, 0.6769202351570129, 0.4581575393676758, 0.5295546054840088, 0.42948853969573975, 0.28497353196144104, 0.4215705990791321, 0.23414918780326843, 0.31588584184646606, 0.4378419518470764, 0.3065018057823181, 0.6458342671394348, 0.4237816631793976, 0.4017394185066223, 0.6789947748184204, 0.4428356885910034, 0.6388193368911743, 0.5486552715301514, 0.47882065176963806, 0.2617994546890259, 0.6517444849014282, 0.42026403546333313, 0.5671502351760864, 0.5002356767654419, 0.608338475227356, 0.4271961450576782, 0.5110663175582886, 0.40036940574645996, 0.5487987399101257, 0.4656513035297394, 0.3869958519935608, 0.46301382780075073, 0.49891549348831177, 0.39406248927116394, 0.45329466462135315, 0.6318700313568115, 0.4937151074409485, 0.35572242736816406, 0.27209755778312683, 0.5468810200691223, 0.40409547090530396, 0.4481413662433624, 0.505862832069397, 0.41193249821662903, 0.4634479582309723, 0.6157476305961609, 0.45787379145622253, 0.23174503445625305, 0.4378930926322937, 0.6456497311592102, 0.40045666694641113, 0.6095210313796997, 0.5695332884788513, 0.49950170516967773, 0.6536673307418823, 0.4210359752178192, 0.3990972936153412, 0.39961859583854675, 0.5350653529167175, 0.427217036485672, 0.4155794084072113, 0.6154018640518188, 0.4908476173877716, 0.7005192041397095, 0.6303793787956238, 0.5217309594154358, 0.6240602135658264, 0.4517726004123688, 0.4396645426750183, 0.6137194037437439, 0.4659545421600342, 0.6620810031890869, 0.44543811678886414, 0.5167111754417419, 0.6468461155891418, 0.46166858077049255, 0.5459617376327515, 0.5495742559432983, 0.4607062041759491, 0.3598610758781433, 0.3078165650367737, 0.38059481978416443, 0.5785108804702759, 0.19922900199890137, 0.5044118762016296, 0.4317927956581116, 0.2994609475135803, 0.6078442335128784, 0.6321358680725098, 0.53342205286026, 0.6036565899848938, 0.474070280790329, 0.429572194814682, 0.4703156650066376, 0.42705443501472473, 0.5398880839347839, 0.6630998849868774, 0.5635955929756165, 0.40198421478271484, 0.42438989877700806, 0.4325372576713562, 0.5535241365432739, 0.47663044929504395, 0.4288228750228882, 0.39683449268341064, 0.5451647043228149, 0.5648629665374756, 0.5037248134613037, 0.48428773880004883, 0.41754111647605896, 0.5652141571044922, 0.5188488960266113, 0.5400396585464478, 0.5253533124923706, 0.524720311164856, 0.40002965927124023, 0.5750817656517029, 0.46880728006362915, 0.6512975096702576, 0.3642416000366211, 0.49856019020080566, 0.37851715087890625, 0.29397052526474, 0.5475768446922302, 0.4085361361503601, 0.460872620344162, 0.3540167212486267, 0.598694384098053, 0.4490131735801697, 0.5801396369934082, 0.6239017844200134, 0.6561968922615051, 0.5749949812889099, 0.24697133898735046, 0.32268816232681274, 0.309826523065567, 0.4663528501987457, 0.4591023623943329, 0.31588584184646606, 0.5375043153762817, 0.503874659538269, 0.5340389013290405, 0.4592468738555908, 0.5939706563949585, 0.24604344367980957, 0.3350537121295929, 0.5322193503379822, 0.3606147766113281, 0.6456471681594849, 0.48668137192726135, 0.5179134607315063, 0.37210774421691895, 0.37480464577674866, 0.39812207221984863, 0.33414745330810547, 0.4288228750228882, 0.5978682041168213, 0.6491960287094116, 0.4632667899131775, 0.413347989320755, 0.31503432989120483, 0.5154446959495544, 0.2728484272956848, 0.4274027943611145, 0.36707475781440735, 0.44793206453323364, 0.45776769518852234, 0.3352292478084564, 0.31758832931518555, 0.5488854050636292, 0.5544484853744507, 0.5100131630897522, 0.5838296413421631, 0.6044108271598816, 0.42086926102638245, 0.5591877102851868, 0.30441802740097046, 0.6630133986473083, 0.5776177048683167, 0.330257385969162, 0.4169157147407532, 0.3755739629268646, 0.6253854036331177, 0.5674793720245361, 0.5286183953285217, 0.43131372332572937, 0.553191602230072, 0.5616980791091919, 0.5267989635467529, 0.3221757411956787, 0.4054652154445648, 0.4850441813468933, 0.47270533442497253, 0.6247143149375916, 0.5267094969749451, 0.6915310025215149, 0.5678216814994812, 0.6172229051589966, 0.48597145080566406, 0.412489652633667, 0.5320343971252441, 0.3762427568435669, 0.6629989147186279, 0.28364983201026917, 0.45564714074134827, 0.5405278205871582, 0.6287368535995483, 0.44568854570388794, 0.3997385799884796, 0.47159332036972046, 0.3732074797153473, 0.313967764377594, 0.2877010405063629, 0.27544185519218445, 0.4963933527469635, 0.363929808139801, 0.33306679129600525, 0.5408790707588196, 0.34000158309936523, 0.3357471227645874, 0.6329755783081055, 0.4378419518470764, 0.4791046977043152, 0.4215705990791321, 0.4350076913833618, 0.4833343029022217, 0.42185837030410767, 0.4279730021953583, 0.6344735622406006, 0.49536752700805664, 0.45108261704444885, 0.6382155418395996, 0.44149768352508545, 0.43776336312294006, 0.3961578905582428, 0.6326762437820435, 0.3863794207572937, 0.2995052933692932, 0.4748394787311554, 0.5439993143081665, 0.299557626247406, 0.6268242597579956, 0.46287545561790466, 0.37484630942344666, 0.3619803190231323, 0.6066911816596985, 0.3913176953792572, 0.6557435989379883, 0.2791988253593445, 0.3622179925441742, 0.6629989147186279, 0.510394275188446, 0.5955976843833923, 0.6105188131332397, 0.37761828303337097, 0.36825719475746155, 0.4874754250049591, 0.5732865333557129, 0.4065820574760437, 0.5744635462760925, 0.5097343921661377, 0.5549454092979431, 0.5174342393875122, 0.33859574794769287, 0.6491233110427856, 0.4707392454147339, 0.34477469325065613, 0.3474280536174774, 0.48662325739860535, 0.48906031250953674, 0.4122683107852936, 0.4978336691856384, 0.5067703723907471, 0.5439350605010986, 0.39691296219825745, 0.5873812437057495, 0.46586763858795166, 0.4574499726295471, 0.5716654658317566, 0.6457815170288086, 0.44226372241973877, 0.2678028345108032, 0.42944395542144775, 0.5135904550552368, 0.4633120596408844, 0.7014954090118408, 0.6312843561172485, 0.5639140605926514, 0.35855814814567566, 0.4708824157714844, 0.5667515397071838, 0.3911440968513489, 0.5246017575263977, 0.4373946487903595, 0.5647965669631958, 0.41580256819725037, 0.4150310754776001];
DTPreds = [];
SVMLinearPreds = SVMPreds(:,1);
SVMRBFPreds = SVMPreds(:,2);
SVMPolynomialPreds = SVMPreds(:,3);

modelPreds = [ANNPreds DTPreds SVMLinearPreds SVMRBFPreds SVMPolynomialPreds];
models = ["ANN" "DecisionTree" "SVMLinear" "SVMRBF" "SVMPolynomial"];
numModels = length(models);
count = 1;
pairing = [];
ttest2Results = zeros(3,10);

% calculate accuracy of all the predictions
% if predictions match their labels
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

%% function to calculate accuracy (for ttest2)
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
function [preds, acc] = evaluateSVM(model, features, labels)

preds = predict(model,features);

labs = table2array(labels);

acc = sqrt(immse(preds, labs));

end


