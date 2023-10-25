function [Centers, betas, Theta,X_activ] = LeaptrainRBFNSOM(X_train, y_train, centersPerCategory, verbose)%,X_activ add
% TRAINRBFN Builds an RBF Network from the provided training set.
%   [Centers, betas, Theta] = trainRBFN(X_train, y_train, centersPerCategory, verbose)
%
%   There are three main steps to the training process:
%     1. Prototype selection through k-means clustering.
%     2. Calculation of beta coefficient (which controls the width of the
%        RBF neuron activation function) for each RBF neuron.
%     3. Training of output weights for each category using gradient descent.
%
%   Parameters
%     X_train  - The training vectors, one per row
%     y_train  - The category values for the corresponding training vector.
%                Category values should be continuous starting from 1. (e.g.,
%                1, 2, 3, ...)
%     centersPerCategory - How many RBF centers to select per category. k-Means
%                          requires that you specify 'k', the number of
%                          clusters to look for.
%     verbose  - Whether to print out messages about the training status.
%
%   Returns
%     Centers  - The prototype vectors stored in the RBF neurons.
%     betas    - The beta coefficient for each coressponding RBF neuron.
%     Theta    - The weights for the output layer. There is one row per neuron
%                and one column per output node / category.

% $Author: ChrisMcCormick $    $Date: 2014/08/18 22:00:00 $    $Revision: 1.3 $

% Get the number of unique categories in the dataset.
%numCats = size(unique(y_train), 1);
numCats =10;

addpath('E:\old matconvnet\SOM-master');
% Set 'm' to the number of data points.
m = size(X_train, 1);
%-----------------------------------------------------------------------------------------------------------
% Ensure category values are non-zero and continuous.
% This allows the index of the output node to equal its category (e.g.,
% the first output node is category 1).
if (any(y_train == 0) || any(y_train > numCats))
    error('Category values must be non-zero and continuous.');
end

% % ================================================
% %       Select RBF Centers and Parameters  Kmeans
% % ================================================
% % Here I am selecting the cluster centers using k-Means clustering.
% % I've chosen to separate the data by category and cluster each
% % category separately, though I've read that this step is often done
% % over the full unlabeled dataset. I haven't compared the accuracy of
% % the two approaches.
%
% if (verbose)
%     disp('1. Selecting centers through k-Means.');
% end
%
% Centers = [];
% betas = [];
% init_Centroids =[];
% % For each of the categories...
% for (c = 1 : numCats)
%
%     if (verbose)
%         fprintf('  Category %d centers...\n', c);
%         if exist('OCTAVE_VERSION') fflush(stdout); end;
%     end
%
%     % Select the training vectors for category 'c'.
%     Xc = X_train((y_train == c), :);
%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% for 0.5
%     centersPerCategory=round(size(Xc,1)/120);%100
%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% for 0.75
%     %cent=round(size(Xc,1)/64);
%     %centersPerCategory=round(size(Xc,1)-cent);



%         init_Centroids = Xc(1:centersPerCategory, :);%%%%%%%%%%%%%%%%%%%%%%%% original

%     % Run k-means clustering, with at most 100 iterations.
%     [Centroids_c, memberships_c] = kMeans(Xc, init_Centroids, 1000);

% ================================================
%      Select RBF Centers and Parameters SOM
% ================================================
% Here I am selecting the cluster centers using k-Means clustering.
% I've chosen to separate the data by category and cluster each
% category separately, though I've read that this step is often done
% over the full unlabeled dataset. I haven't compared the accuracy of
% the two approaches.


if (verbose)
    disp('1. clustering SOM.');
end

Centers = [];
betas = [];
init_Centroids =[];
train_classlabel=y_train';
%p = length(Xc(:,1));

% Number of iterations
N =4;
M = 4;
nb_neurons = N * M;
nb_iter = 10;
% Effective width
eff_width_init = (sqrt(N*N + M*M))/2;
eff_width_time_cst  = nb_iter / log(eff_width_init);
% Learning rate
learning_rate_init = 0.0001;
learning_rate_time_cst = nb_iter;

Centroids_c=[];
for (c = 1 : numCats)
    Centroids=[];
    memberships_c=[];
    % Use the SOM function
    % Select the training vectors for category 'c'.
    fprintf('  Category %d centers...\n', c);
    Xc = X_train((y_train == c), :);
    % ==============================================
    %      Find initial cluster centers
    % ===============================================
    
    % Pick the first 'centersPerCategory' samples to use as the initial
    % centers.
    centersPerCategory=nb_neurons;
    init_Centroids =[];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   add
    centersPerCategory1=randperm(size(Xc,1));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   add
    for w=1:centersPerCategory%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   add
        init_Centroids =[init_Centroids;Xc(centersPerCategory1(w),:)];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   add
    end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   add
    
    [train_weights] = SOM1(init_Centroids',Xc', N, M, nb_iter, eff_width_init, eff_width_time_cst, learning_rate_init,learning_rate_time_cst);
    % Define the label vector
    train_SOM_labels = zeros(nb_neurons,1);
    
    %--------------------------------- Loop on the neurons of the SOM to predict cluster centers
    for j = 1:nb_neurons
        
        % Define the current neuron
        current_neuron = train_weights(:,j);
        
        % Determine the winner input signal for neuron j (the closest one)
        [winner_input_sample, winner_input_idx, winner_input_distance] = find_winner(Xc', current_neuron);
        
        % Set the label of the neuron j
        train_SOM_labels(j) = train_classlabel(winner_input_idx);
        
        % Centroids
        Centroids =[Centroids;winner_input_sample'];  %%%%%%%ADDDDDDDDDDDDDD
    end
    %-------------------------- Assign each sample to closest cluster center
    nb_test_samples = length(Xc(:,1));
    test_data=Xc';
    % Define the label vector of the test data
    test_SOM_label = zeros(nb_test_samples,1);
    test_win=[];%%%%%%%%%%%%%%%%%%%%%%%%%% save feature Test
    % Loop on the test samples
    for i = 1:nb_test_samples
        
        current_test_sample = test_data(:,i);
        
        % Determine the winner neuron : the closest neuron of the SOM from the
        % current test sample
        [winner_output_neuron, winner_output_idx, winner_output_distance] = find_winner(train_weights, current_test_sample);
        
        % Label the test sample with the one corresponding to the winner neuron
        try
            test_SOM_label(i) = winner_output_idx;
        catch
            test_SOM_label(i)
        end
        %test_win=[test_win;winner_output_neuron'];%%%%%% save feature Test
    end
    memberships_c= test_SOM_label;
    
    Centroids_c=Centroids;
    
    %-------------------------------End SOM------------------------------
    %------------------------------------------------------------------
    
    % Remove any empty clusters.
    toRemove = [];
    labcenter= [];
    % For each of the centroids...
    for (i = 1 : size(Centroids_c, 1))
        % If this centroid has no members, mark it for removal.
        if ((sum(memberships_c == i) == 0)| (sum(memberships_c == i) <= 1))
            toRemove = [toRemove; i];
        else
            labcenter=[labcenter;i];
        end
    end
    
    % If there were empty clusters...
    if (~isempty(toRemove))
        % Remove the centroids of the empty clusters.
        Centroids_c(toRemove, :) = [];
        
        % Reassign the memberships (index values will have changed).
        %memberships_c = findClosestCentroids(Xc, Centroids_c);
    end
    
    % ================================
    %    Compute Beta Coefficients
    % ================================
    if (verbose)
        fprintf('  Category %d betas...\n', c);
        if exist('OCTAVE_VERSION') fflush(stdout); end;
    end
    
    % Compute betas for all the clusters.
    betas_c = computeRBFBetas1(Xc, Centroids_c,labcenter, memberships_c);
    
    
    % Add the centroids and their beta values to the network.
    Centers = [Centers; Centroids_c];
    betas = [betas; betas_c];
end
%--------------------------------------------------------------------------------------------------------------
% Get the final number of RBF neurons.
numRBFNeurons = size(Centers, 1);
% ===================================
%        Train Output Weights
% ===================================

% ==========================================================
%       Compute RBF Activations Over The Training Set
% ===========================================================
if (verbose)
    disp('2. Calculate RBF neuron activations over full training set.');
end


% First, compute the RBF neuron activations for all training examples.

% The X_activ matrix stores the RBF neuron activation values for each
% training example: one row per training example and one column per RBF
% neuron.
X_activ = zeros(m, numRBFNeurons);

% For each training example...
for (i = 1 : m)
    
    input = X_train(i, :);
    
    % Get the activation for all RBF neurons for this input.
    z = getRBFActivations(Centers, betas, input);
    
    % Store the activation values 'z' for training example 'i'.
    X_activ(i, :) = z';
end

% Add a column of 1s for the bias term.
X_activ = [ones(m, 1), X_activ];

% =============================================
%        Learn Output Weights
% =============================================

if (verbose)
    disp('3. Learn output weights.');
end

% Create a matrix to hold all of the output weights.
% There is one column per category / output neuron.
Theta = zeros(numRBFNeurons + 1, numCats);

% For each category...
for (c = 1 : numCats)
    
    % Make the y values binary--1 for category 'c' and 0 for all other
    % categories.
    y_c = (y_train == c);
    
    % Use the normal equations to solve for optimal theta.
    try
        Theta(:, c) = pinv(X_activ' * X_activ) * X_activ' * y_c;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   y_c'
    catch
        Theta(:, c);
    end
end

end
