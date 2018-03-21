%% Block Coordinate Descent (BCD) Algorithm for Training DNNs (3-layer MLP) (CIFAR-10 dataset)
%%% 5 runs, seed = 10, 20, 30, 40, 50; validation accuracies: 
%%% (alpha = 5) 0.4499, 0.4519, 0.4484, 0.4496, 0.4489
clear all
close all
clc

addpath Algorithms Tools

disp("MLP with Three Hidden Layers using the CIFAR-10 dataset (Jinshan's Algorithm)")

rng('default');
seed = 40;
rng(seed);
fprintf('Seed = %d \n', seed)


% read in CIFAR-10 dataset into Matlab format
addpath cifar-10-batches-mat
disp('Reading in CIFAR-10 training dataset')
data_batch_1 = load('data_batch_1.mat');
data_batch_2 = load('data_batch_2.mat');
data_batch_3 = load('data_batch_3.mat');
data_batch_4 = load('data_batch_4.mat');
data_batch_5 = load('data_batch_5.mat');

% train data and labels
x_train = [data_batch_1.data;data_batch_2.data;data_batch_3.data;data_batch_4.data;data_batch_5.data];
y_train = [data_batch_1.labels;data_batch_2.labels;data_batch_3.labels;data_batch_4.labels;data_batch_5.labels];
clear data_batch_1 data_batch_2 data_batch_3 data_batch_4 data_batch_5
disp('Done!')

x_train = reshape(permute(reshape(x_train,[50000 32 32 3]),[1,3,2,4]),[50000 3072])';
% disp(x_train(1,1:10))
%% Visual training data samples
% figure;
% for i = 1:5
%     subplot(1,5,i)
%     img{i} = reshape(x_train(:,i),[32 32 3]);
%     imshow(img{i})
% end
% 
% close all

%% Extract Classes for the training set
num_classes = 10; % choose the first num_class classes in the CIFAR-10 dataset for training
X = [y_train';x_train];
[~,col] = find(X(1,:) < num_classes);
X = X(:,col);
[~,N] = size(X);
% X = X(:,randperm(N)); % shuffle the training dataset
x_train = X(2:end,:);
y_train = X(1,:)';
clear X

x_train = double(x_train); y_train = double(y_train);
y_one_hot = ind2vec((y_train'+1));
[K,N] = size(y_one_hot);
[d,~] = size(x_train);

%% Test data
% read in test data and labels
disp('Reading in CIFAR-10 test dataset')
test_batch = load('test_batch.mat');
x_test = test_batch.data; % test data
y_test = test_batch.labels; % labels
clear test_batch
disp('Done!')

x_test = reshape(permute(reshape(x_test,[10000 32 32 3]),[1,3,2,4]),[10000 3072])';

%% Visual test data samples
% figure;
% for i = 1:5
%     subplot(1,5,i)
%     img{i} = reshape(x_test(:,i),[32 32 3]);
%     imshow(img{i})
% end
% 
% close all

%% Extract Classes for the test set
X_test = [y_test';x_test];
[~, col_test] = find(X_test(1,:) < num_classes);
X_test = X_test(:,col_test);
[~,N_test] = size(X_test);
% X_test = X_test(:,randperm(N_test,N_test)); % shuffle the test dataset
x_test = X_test(2:end,:);
y_test = X_test(1,:)';
clear X_test

x_test = double(x_test); y_test = double(y_test);
y_test_one_hot = ind2vec((y_test'+1));
[~,N_test] = size(y_test_one_hot);

%% Data Transformation
x_train = x_train/255;
x_test = x_test/255;

%% GPU Array
% x_train = gpuArray(x_train);
% y_train = gpuArray(y_train);
% x_test = gpuArray(x_test);
% y_test = gpuArray(y_test);

%% Main Algorithm (Jinshan's Algorithm)
% Initialization of parameters 
d0 = d; d1 = 4e3; d2 = 1e3; d3 = 4e3; d4 = K; % Layers: input + 3 hidden + output

W1 = 0.01*randn(d1,d0); b1 = 0.1*ones(d1,1); 

W2 = 0.01*randn(d2,d1); b2 = 0.1*ones(d2,1); 

W3 = 0.01*randn(d3,d2); b3 = 0.1*ones(d3,1); 

% W4 = 0.01*sprandn(d4,d3,0.1); b4 = zeros(d4,1); 
W4 = 0.01*randn(d4,d3); b4 = 0.1*ones(d4,1); 


indicator = 1; % 0 = sign; 1 = ReLU; 2 = tanh; 3 = sigmoid

switch indicator
    case 0 % sign (binary)
        U1 = W1*x_train+b1; V1 = sign(U1); 
        U2 = W2*V1+b2; V2 = sign(U2); 
        U3 = W3*V2+b3; V3 = sign(U3); 
        U4 = W4*V3+b4; V4 = U4;
	case 1 % ReLU
        U1 = W1*x_train+b1; V1 = max(0,U1); 
        U2 = W2*V1+b2; V2 = max(0,U2); 
        U3 = W3*V2+b3; V3 = max(0,U3);  
        U4 = W4*V3+b4; V4 = U4;
	case 2 % hard tanh
        U1 = W1*x_train+b1; V1 = tanh_proj(U1); 
        U2 = W2*V1+b2; V2 = tanh_proj(U2); 
        U3 = W3*V2+b3; V3 = tanh_proj(U3);
        U4 = W4*V3+b4; V4 = U4;
	case 3 % hard sigmoid
        U1 = W1*x_train+b1; V1 = sigmoid_proj(U1); 
        U2 = W2*V1+b2; V2 = sigmoid_proj(U2); 
        U3 = W3*V2+b3; V3 = sigmoid_proj(U3);
        U4 = W4*V3+b4; V4 = U4;
end

% lambda = 0;
gamma = 0.75; %0.7
gamma1 = gamma; gamma2 = gamma; gamma3 = gamma; gamma4 = gamma;

rho = gamma;
rho1 = rho; rho2 = rho; rho3 = rho; rho4 = rho;

% alpha1 = 10; 
alpha1 = 5; 
alpha = 5;
alpha2 = alpha; alpha3 = alpha; alpha4 = alpha; 
alpha5 = alpha; alpha6 = alpha; alpha7 = alpha; 
alpha8 = alpha; % alpha9 = alpha; alpha10 = alpha; 

% beta = 0.95;
% beta1 = beta; beta2 = beta; beta3 = beta; beta4 = beta; 
% beta5 = beta; beta6 = beta; beta7 = beta; 
% beta8 = beta; beta9 = beta; beta10 = beta; 

% t = 0.1;

% niter = input('Number of iterations: ');
niter = 50;
loss1 = zeros(niter,1);
loss2 = zeros(niter,1);
layer1 = zeros(niter,1);
layer2 = zeros(niter,1);
layer3 = zeros(niter,1);
layer4 = zeros(niter,1);
layer11 = zeros(niter,1);
layer21 = zeros(niter,1);
layer31 = zeros(niter,1);
layer41 = zeros(niter,1);
accuracy_train = zeros(niter,1);
accuracy_test = zeros(niter,1);
time1 = zeros(niter,1);

% Iterations
for k = 1:niter
    tic
    
    % record previous W1, W2, W3, W4, b1, b2, b3, b4
    W10 = W1; W20 = W2; W30 = W3; W40 = W4;
    b10 = b1; b20 = b2; b30 = b3; b40 = b4;
    
    % update V4
    V4 = (y_one_hot + gamma4*U4 + alpha1*V4)/(1+gamma4+alpha1);
    
    % update U4 
    U4 = (gamma4*V4+rho4*(W4*V3+b4))/(gamma4+rho4);
    
    % update W4 and b4
    [W4,b4] = updateWb_js(U4,V3,W4,b4,alpha2,rho4);
%     [W4,b4] = updateWb_js_2(U4,V3,W4,b4);
    
    % update V3
    V3 = updateV_js(U3,U4,W4,b4,rho4,gamma3,indicator);
    
    % update U3
    U3 = relu_prox(V3,(rho3*(W3*V2+b3)+alpha3*U3)/(rho3+alpha3),(rho3+alpha3)/gamma3,d3,N);
%     U3 = relu_prox2(V3,(rho3*(W3*V2+b3)+alpha3*U3)/(rho3+alpha3),(rho3+alpha3)/gamma3,d3,N);
    
    % update W3 and b3
    [W3,b3] = updateWb_js(U3,V2,W3,b3,alpha4,rho3);
%     [W3,b3] = updateWb_js_2(U3,V2,W3,b3);
    
    % update V2
    V2 = updateV_js(U2,U3,W3,b3,rho3,gamma2,indicator);
    
    % update U2
    U2 = relu_prox(V2,(rho2*(W2*V1+b2)+alpha5*U2)/(rho2+alpha5),(rho2+alpha5)/gamma2,d2,N);
%     U2 = relu_prox2(V2,(rho2*(W2*V1+b2)+alpha5*U2)/(rho2+alpha5),(rho2+alpha5)/gamma2,d2,N);
    
    % update W2 and b2
    [W2,b2] = updateWb_js(U2,V1,W2,b2,alpha6,rho2);
%     [W2,b2] = updateWb_js_2(U2,V1,W2,b2);
    
    % update V1
    V1 = updateV_js(U1,U2,W2,b2,rho2,gamma1,indicator);
    
    % update U1
    U1 = relu_prox(V1,(rho1*(W1*x_train+b1)+alpha7*U1)/(rho1+alpha7),(rho1+alpha7)/gamma1,d1,N);
%     U1 = relu_prox2(V1,(rho1*(W1*x_train+b1)+alpha7*U2)/(rho1+alpha7),(rho1+alpha7)/gamma1,d1,N);

    % update W1 and b1
    [W1,b1] = updateWb_js(U1,x_train,W1,b1,alpha8,rho1);
%     [W1,b1] = updateWb_js_2(U1,x_train,W1,b1);
     
    % Training accuracy
    switch indicator
    case 0 % sign
    	a1_train = sign(W1*x_train+b1);
        a2_train = sign(W2*a1_train+b2);
        a3_train = sign(W3*a2_train+b3);
    case 1 % ReLU
        a1_train = max(0,W1*x_train+b1);
        a2_train = max(0,W2*a1_train+b2);
        a3_train = max(0,W3*a2_train+b3);
    case 2 % tanh
        a1_train = tanh_proj(W1*x_train+b1);
        a2_train = tanh_proj(W2*a1_train+b2);
        a3_train = tanh_proj(W3*a2_train+b3);
    case 3 % sigmoid
        a1_train = sigmoid_proj(W1*x_train+b1);
        a2_train = sigmoid_proj(W2*a1_train+b2);
        a3_train = sigmoid_proj(W3*a2_train+b3);
    end
    [~,pred] = max(W4*a3_train+b4,[],1);
    
    % Test accuracy
    switch indicator
        case 0 % sign
        a1_test = sign(W1*x_test+b1);
        a2_test = sign(W2*a1_test+b2);
        a3_test = sign(W3*a2_test+b3);
        case 1 % ReLU
        a1_test = max(0,W1*x_test+b1); 
        a2_test = max(0,W2*a1_test+b2); 
        a3_test = max(0,W3*a2_test+b3);
        case 2 % tanh
        a1_test = tanh_proj(W1*x_test+b1); 
        a2_test = tanh_proj(W2*a1_test+b2); 
        a3_test = tanh_proj(W3*a2_test+b3);
        case 3 % sigmoid
        a1_test = sigmoid_proj(W1*x_test+b1); 
        a2_test = sigmoid_proj(W2*a1_test+b2); 
        a3_test = sigmoid_proj(W3*a2_test+b3);
    end
    [~,pred_test] = max(W4*a3_test+b4,[],1);
    
    
    loss1(k) = gamma4/2*norm(V4-y_one_hot,'fro')^2;
    loss2(k) = loss1(k)+rho1/2*norm(W1*x_train+b1-U1,'fro')^2+rho2/2*norm(W2*V1+b2-U2,'fro')^2+rho3/2*norm(W3*V2+b3-U3,'fro')^2+rho4/2*norm(W4*V3+b4-U4,'fro')^2;
    loss2(k) = loss2(k)+gamma1/2*norm(V1-max(U1,0),'fro')^2+gamma2/2*norm(V2-max(U2,0),'fro')^2+gamma3/2*norm(V3-max(U3,0),'fro')^2+gamma4/2*norm(V4-U4,'fro')^2;
 
    % speed of learning 
    layer1(k) = norm(W1-W10,'fro');
    layer2(k) = norm(W2-W20,'fro');
    layer3(k) = norm(W3-W30,'fro');
    layer4(k) = norm(W4-W40,'fro');
        
    accuracy_train(k) = sum(pred'-1 == y_train)/N;
    accuracy_test(k) = sum(pred_test'-1 == y_test)/N_test;
    time1(k) = toc;
    fprintf('epoch: %d, squared loss: %f, total loss: %f, training accuracy: %f, validation accuracy: %f\n',k,loss1(k),loss2(k),accuracy_train(k),accuracy_test(k))
    fprintf('speed of learning: HL1: %f; HL2: %f; HL3: %f; OL: %f; time: %f\n',layer1(k),layer2(k),layer3(k),layer4(k),time1(k))
end


fprintf('squared error: %f\n',loss1(k))
fprintf('sum of inter-layer loss: %f\n',loss2(k)-loss1(k))
%disp(full(cross_entropy(y_one_hot,a2,V,c)))


%% Plots
figure;
graph1 = semilogy(1:niter,loss1,1:niter,loss2);
set(graph1,'LineWidth',1.5);
l1 = legend('Squared loss','Total loss');
% l1.Interpreter = 'latex';
ylabel('Loss')
xlabel('Epochs')
title('Three-layer MLP')

figure;
graph2 = semilogy(1:niter,accuracy_train,1:niter,accuracy_test);
set(graph2,'LineWidth',1.5);
% ylim([0.85 1])
l2 = legend('Training accuracy','Validation accuracy','Location','southeast');
% l2.Interpreter = 'latex';
ylabel('Accuracy')
xlabel('Epochs')
title('Three-layer MLP')

figure;
graph3 = semilogy(1:niter,layer1,1:niter,layer2,1:niter,layer3,1:niter,layer4);
set(graph3,'LineWidth',1.5);
l3 = legend('Hidden layer 1','Hidden layer 2','Hidden layer 3','Output layer','Location','northeast');
l3.Interpreter = 'latex';
ylabel('$\|W^{k}-W^{k-1}\|_F$','interpreter','latex')
xlabel('Epochs','interpreter','latex')
title('Speed of learning: Three-layer MLP','interpreter','latex')

% figure;
% graph4 = semilogy(1:niter,layer11,1:niter,layer21,1:niter,layer31,1:niter,layer41);
% set(graph4,'LineWidth',1.5);
% l4 = legend('Hidden layer 1','Hidden layer 2','Hidden layer 3','Output layer','Location','northeast');
% l4.Interpreter = 'latex';
% ylabel('$\nabla_{b^{k}}\bar{\mathcal{L}}$','interpreter','latex')
% xlabel('Epochs','interpreter','latex')
% title('Speed of learning: Three-layer MLP','interpreter','latex')
%% Training error
switch indicator
    case 1 % ReLU
        a1_train = max(0,W1*x_train+b1);
        a2_train = max(0,W2*a1_train+b2);
        a3_train = max(0,W3*a2_train+b3);
    case 2 % tanh
        a1_train = tanh_proj(W1*x_train+b1);
        a2_train = tanh_proj(W2*a1_train+b2);
        a3_train = tanh_proj(W3*a2_train+b3);
    case 3 % sigmoid
        a1_train = sigmoid_proj(W1*x_train+b1);
        a2_train = sigmoid_proj(W2*a1_train+b2);
        a3_train = sigmoid_proj(W3*a2_train+b3);
end

[~,pred] = max(W4*a3_train+b4,[],1);
pred_one_hot = ind2vec(pred);
accuracy_final = sum(pred'-1 == y_train)/N;
fprintf('Training accuracy using output layer: %f\n',accuracy_final);
% error = full(norm(pred_one_hot-y_one_hot,'fro')^2/(2*N));
% fprintf('Training MSE using output layer: %f\n',error);

%% Test errors
switch indicator
    case 1 % ReLU
        a1_test = max(0,W1*x_test+b1); 
        a2_test = max(0,W2*a1_test+b2); 
        a3_test = max(0,W3*a2_test+b3); 
    case 2 % tanh
        a1_test = tanh_proj(W1*x_test+b1); 
        a2_test = tanh_proj(W2*a1_test+b2); 
        a3_test = tanh_proj(W3*a2_test+b3);
    case 3 % sigmoid
        a1_test = sigmoid_proj(W1*x_test+b1); 
        a2_test = sigmoid_proj(W2*a1_test+b2); 
        a3_test = sigmoid_proj(W3*a2_test+b3);
end


[~,pred_test] = max(W4*a3_test+b4,[],1);
pred_test_one_hot = ind2vec(pred_test);
accuracy_test_final = sum(pred_test'-1 == y_test)/N_test;
fprintf('Test accuracy using output layer: %f\n',accuracy_test_final);
% error_test = full(norm(pred_test_one_hot-y_test_one_hot,'fro')^2/(2*N_test));
% fprintf('Test MSE using output layer: %f\n',error_test);

%% Linear SVM for train errors
% rng(seed); % For reproducibility
% SVMModel = fitcecoc(a3_train,y_train,'ObservationsIn','columns');
% L = resubLoss(SVMModel,'LossFun','classiferror');
% % fprintf('Training error classified with SVM: %f\n',L);
% fprintf('Training accuracy classified with SVM: %f\n',1-L);

%% SVM test error
% predictedLabels = predict(SVMModel,a3_test,'ObservationsIn','columns');
% accuracy = sum(predictedLabels==y_test)/numel(predictedLabels);
% fprintf('Test accuracy classified with SVM: %f\n',accuracy);


%% Toolbox training

% layers = [imageInputLayer([28 28 1]);
%           fullyConnectedLayer(100);
%           reluLayer();
%           fullyConnectedLayer(K);
%           softmaxLayer();
%           classificationLayer()];
%       
%       
% options = trainingOptions('sgdm','ExecutionEnvironment','gpu','MaxEpochs',50,'InitialLearnRate',0.01);
% 
% rng(20)
% net = trainNetwork(reshape(x_train,28,28,1,N),categorical(y_train),layers,options);
% 
% % Test accuracy
% YTest = classify(net,reshape(x_test,28,28,1,N_test));
% TTest = categorical(y_test);
% accuracy1 = sum(YTest == TTest)/numel(TTest);  
% fprintf('Test accuracy with backprop: %f\n',accuracy1);

%% Feature extraction + SVM + Test accuracy
% trainFeatures = activations(net,reshape(x_train,28,28,1,N),3);
% svm = fitcecoc(trainFeatures,categorical(y_train));
% L2 = resubLoss(svm,'LossFun','classiferror');
% fprintf('Training error using backprop classified with SVM: %f\n',L2);
% fprintf('Training accuracy using backprop classified with SVM: %f\n',1-L2);
% 
% testFeatures = activations(net,reshape(x_test,28,28,1,N_test),3);
% testPredictions = predict(svm,testFeatures);
% accuracy2 = sum(categorical(y_test) == testPredictions)/numel(categorical(y_test));
% fprintf('Test accuracy using backprop classified with SVM: %f\n',accuracy2);

