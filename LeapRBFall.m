function LeapRBFall(varargin)%------------all_trainRBF

close all, clear all, clc
addpath('E:\old matconvnet\rbfn_O\kMeans');
addpath('E:\old matconvnet\rbfn_O\RBFN');
addpath('E:\old matconvnet\rbfn_O');

ground=[];
pred=[];
lambda=1;
numCats=10;
opts.numEpochs=10;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9;

%load('E:\Matlab\TrainF\TTR');%%%%%%%%%%%%%%%%%% 3dim
%load('E:\Matlab\TrainF\TT');%%%%%%%%%%%%%%%%%% 3dim
load('E:\Matlab\LeapGesture\Trainfeature');%%%%%%%%%%%%%%%%%% 1dim
im=im';
labels=label;
%load('E:\Matlab\TrainF\Rnn(feature(ch(Single+rgb)+RBFN+Som))-Lstm');%Lstm features
%im=Fr2;

%centersPerCategory=(20*20)/20;%N*M/No.class   %SOM
centersPerCategory=round(size(im,1)/50);      %RBF   
numRBFNeurons=centersPerCategory*numCats;
Theta = zeros(numRBFNeurons + 1, numCats);
betass=zeros(numRBFNeurons,1);
Centerss=zeros(numRBFNeurons,2048);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%% Randomize the input %%%%%%%%%%%%%%%%%%%%%%%%
%     train = randperm(size(TT,1)) ;
%     labels1=[]; TT1=[];
%     for i=1:size(TT,1)
%       TT1=[TT1;TT(train(i),:)];
%       labels1(i)=labels(train(i));
%     end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55555555%%%%%%%%%%%%%%%%%%%


%for epoch=1:opts.numEpochs



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RBF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Centerss, betass, Theta,X_activ] = trainRBFN11(TT1, labels1', centersPerCategory,Centerss, betass,true);
%     [Centerss, betass, Theta,X_activ] = trainRBFN(TT1, labels1', centersPerCategory,true);%3d

%[Centerss, betass, Theta,X_activ] = LeaptrainRBFN(im, labels', centersPerCategory,true);
[Centerss, betass, Theta,X_activ] = LeaptrainRBFNSOM(im, labels', centersPerCategory,true);
% Thetaa = zeros(size(Theta,1),size(Theta,2));
% momentum = zeros(size(Theta,1),size(Theta,2));
% for epoch1=1:20
%     %
%     Theta=Theta+Thetaa;
%     %
%     %         %cost = vl_nnsoftmaxloss(h, labels) ;%%%%%%%%%%%%%%%%%% softmax
%     %         %dzxx = vl_nnsoftmaxloss(h, labels, cost) ;%%%%%%%%%%%%%%%%%% softmax
%     %         %dzxx=reshape(dzxx,[size(dzxx,3) size(dzxx,4)]);%%%%%%%%%%%%%%%%%% softmax
%     %         %dzxx=dzxx*X_activ;%%%%%%%%%%%%%%%%%% softmax
%     %         %dzxx=dzxx';%%%%%%%%%%%%%%%%%% softmax
     [cost, dzxx] = costFunctionRBFN(Theta, X_activ, labels', lambda);
%     %         % [Theta, momentum] = accumulate_gradients(opts, learningRate, batchSize, Theta, momentum,dzdxx) ;
%     %         %
%     %         %
%     %         % ========================================
%     %         %       Measure Grediant
%     %         % ========================================
%     %
%     thisDecay = opts.weightDecay ;
%     thisLR = opts.learningRate  ;
%     momentum = opts.momentum * momentum ...
%         - thisDecay -dzxx ;% (1 / batchSize) * dzxx ;
%     Theta = Theta +thisLR *momentum ;
%     %
% 
% end
% 
% 
% figure
% RBF_label_categories = categorical(labels,[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20],{'1', '2', '3', '4', '5', '6', '7', '8', '9','10', '11', '12', '13' ,'14', '15', '16', '17', '18', '19', '20'});
% histogram(RBF_label_categories)
% title('Distribution of the labels categories of the RBF')
% ylabel('number of labels')
% 
% 
% disp('Measuring training accuracy...');

numRight = 0;

wrong =0;ground=[];
pred=[];

% For each training sample...
for (i = 1 : size(im, 1))%TT 3d
    % Compute the scores for both categories.
    % = evaluateRBFN(Centerss, betass, Theta, TT(i, :));
    scores = evaluateRBFN(Centerss, betass, Theta, im(i, :));
    [maxScore, category] = max(scores);
    pred(end+1)=category;
    ground(end+1)=labels(i);

    % Validate the result.
    if (pred(i) == labels(i))
        numRight = numRight + 1;
    else
        wrong = wrong+1;
    end

end
%accuracy=(numRight/size(TT1, 1))*100 %%%3d
accuracy=(numRight/size(im, 1))*100


%end
%save('E:\old matconvnet\rbfn_O\Data\LeaptrainRBNF1,50.mat','Centerss', 'betass', 'Theta');
%save('E:\old matconvnet\rbfn_O\Data\LeaptrainRBNFSOM,6.mat','Centerss', 'betass', 'Theta','-v7.3');

%--------------------------------------extract test features(X_activ)
%NNRBFallTf();

% %--------------------------------------------%Test
% 
 
 load('E:\Matlab\LeapGesture\Testfeature');%%%%%%%%%%%%%%%%%% 1dim  for Test
 %load('E:\old matconvnet\rbfn_O\Data\LeaptrainRBNF1,50.mat');%%%%%%%%%%%%%%%%%%for Test
 load('E:\old matconvnet\rbfn_O\Data\LeaptrainRBNFSOM,4.mat');%%%%%%%%%%%%%%%%%%for Test
im=im';
labels=label;
%load('E:\Matlab\Testall7\Rnn(feature(ch(Single+rgb)+RBFN+Som))-Lstm.mat');%Lstm features
% im=Fr2;

disp('Measuring training accuracy...');

numRight = 0;

wrong =0;ground=[];
pred=[];

% For each testing sample...
for (i = 1 : size(im, 1))%TT 3d
    % Compute the scores for both categories.
    
    scores = evaluateRBFN(Centerss, betass, Theta, im(i, :));
    [maxScore, category] = max(scores);
    pred(end+1)=category;
    ground(end+1)=labels(i);
    
    % Validate the result.
    if (pred(i) == labels(i))
        numRight = numRight + 1;
    else
        wrong = wrong+1;
    end
    
end
%accuracy=(numRight/size(TT1, 1))*100 %%%3d
accuracy=(numRight/size(im, 1))*100

% save('E:\old matconvnet\confusionRBF\Leappr1.mat','pred');
% save('E:\old matconvnet\confusionRBF\Leapgt1.mat','ground');
fscore(ground,pred);
%mAp(ground,pred);
ROC(ground,pred);
save('E:\old matconvnet\confusionRBF\LeapSOM,2,pr1.mat','pred');
save('E:\old matconvnet\confusionRBF\LeapSOM,2,gt1.mat','ground');

load('E:\old matconvnet\confusionRBF\Leappr1.mat')
load('E:\old matconvnet\confusionRBF\Leapgt1.mat')


conver(ground,pred);

end
%--------------------------------------
function conver(gt,p)
%--------------------------------------
gt1 = zeros(10,length(gt),'double') ;
p1 = zeros(10,length(gt),'double') ;

for j = 1 : length(gt)
    a = gt(1,j) ;
    gt1(a,j) = 1 ;
end

for j = 1 : length(gt)
    a = p(1,j)  ;
    p1(a,j) = 1 ;
end
plotconfusion(gt1,p1);
end
%--------------------------------------
function fscore(groundTruth,predictions)
%--------------------------------------
% Step 2: Compute Confusion Matrix
numClasses = max(groundTruth); % Number of classes
confusionMatrix = zeros(numClasses, numClasses);

for i = 1:numClasses
    for j = 1:numClasses
        confusionMatrix(i, j) = sum(groundTruth == i & predictions == j);
    end
end

% Step 3: Calculate Precision, Recall, and F1-Score for Each Class
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confusionMatrix(i, i);
    FP = sum(confusionMatrix(:, i)) - TP;
    FN = sum(confusionMatrix(i, :)) - TP;

    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Step 4: Compute Weighted Average F1-Score (Optional)
support = sum(confusionMatrix, 2); % Number of samples for each class
weightedF1Score = sum(f1Score .* support) / sum(support);

% Step 5: Display F1-Scores
disp('F1-Scores for Each Class:');
disp(f1Score);
disp(['Weighted Average F1-Score: ', num2str(weightedF1Score)]);
% Step 6: Display mAp
disp('Ap:');
disp(precision);
disp(['mAp: ', num2str(mean(precision))]);


end
%-----------------
function ap = averagePrecision(labels, scores)
    [~, sortedIdx] = sort(scores, 'descend');
    labels = labels(sortedIdx);
    
    tp = cumsum(labels);
    fp = cumsum(~labels);
    
    recall = tp / sum(labels);
    precision = tp ./ (tp + fp);
    
    ap = sum(precision .* labels) / sum(labels);
end

%----------------
%--------------------------------------
function ROC(groundTruth,predictions)
%--------------------------------------


% Convert the integer ground-truth labels to one-hot encoded vectors
groundTruthOneHot = full(ind2vec(groundTruth));
predictionBinary = full(ind2vec(predictions));

% % Convert the integer prediction labels to a matrix of binary predictions
% numClasses = max(groundTruth);
% numSamples = length(predictions);
% predictionBinary = zeros(numClasses, numSamples);
% for i = 1:numSamples
%     predictionBinary(predictions(i), i) = 1;
% end

% Calculate the ROC curve for each class
[X, Y, ~, AUC] = perfcurve(groundTruthOneHot(:)', predictionBinary(:)', 1);

% Plot the ROC curve
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');



end
%--------------------------------------
function mAp(groundTruth,predictionScores)
%--------------------------------------

% Step 2: Convert Labels to Binary Format
numClasses = max(groundTruth); % Number of classes
binaryLabels = zeros(length(groundTruth), numClasses);

for class = 1:numClasses
    binaryLabels(:, class) = groundTruth == class;
end

% Step 3: Calculate Average Precision (AP) for Each Class
AP = zeros(numClasses, 1);

for class = 1:numClasses
    AP(class) = averagePrecision(binaryLabels(:, class), predictionScores);
end

% Step 4: Calculate Mean Average Precision (mAP)
mAP = mean(AP);

disp(['Mean Average Precision (mAP): ', num2str(mAP)]);
end
