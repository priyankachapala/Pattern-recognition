%FLD

clear all;
close all;
clc;
%% Initialisation and generation of data points

dim = 4;
n1 = 100;
m1 = [0.75 0.5 0.45 0.45];
cov1 = [0.1 0 0 0 ;
        0 0.5 0 0 ;
        0 0 0.75 0;
        0 0 0 0.2];
% rng('default'); 
G1 = mvnrnd(m1,cov1,n1);
n2 = 100;
m2 = [0.1 0.1 0.25 0.2];
cov2 = [0.25 0 0 0 ;
        0 0.1 0 0 ;
        0 0 0.375 0;
        0 0 0 0.25];
% rng('default'); 
G2 = mvnrnd(m2,cov2,n2);
n3 = 100;
m3 = [1.5 1.25 1.75 2];
cov3 = [0.5 0 0 0 ;
        0 0.2 0 0 ;
        0 0 0.45 0;
        0 0 0 0.3];
% rng('default'); 
G3 = mvnrnd(m3,cov3,n3);

%% Separating training and testing data points

train_size1 = 0.8; 
Tr1 = G1(1:train_size1*n1,:);
N1 = length(Tr1);
Test1 = G1((train_size1*n1)+1:n1,:);
train_size2 = 0.8;
Tr2 = G2(1:train_size2*n2,:);
N2 = length(Tr2);
Test2 = G2((train_size2*n2)+1:n2,:);
train_size3 = 0.8; 
Tr3 = G3(1:train_size3*n3,:);
N3 = length(Tr3);
Test3 = G3((train_size3*n3)+1:n3,:);

%% Finding the mean and covariance of the trained points 

m_tr1 = mean(Tr1);
cov_tr1 = cov(Tr1);
m_tr2 = mean(Tr2);
cov_tr2 = cov(Tr2);
m_tr3 = mean(Tr3);
cov_tr3 = cov(Tr3);

%% Finding between class covariance matrix

m = (sum(Tr1+Tr2+Tr3))/(3*length(Tr1));
SB = (N1*(m_tr1 - m)'*(m_tr1- m))+(N2*(m_tr2 - m)'*(m_tr2- m)) +(N3*(m_tr3 - m)'*(m_tr3- m));

%% Finding within class covariance matrix

SW = cov_tr1+cov_tr2+cov_tr3;

%% FLD

M = inv(SW)*SB;

[V, D] =eig(M);

D_temp1 = D;

max1 = D_temp1(1,1);
maxind1 = 1;
for i=1:length(D_temp1)
    if D_temp1(i,i) > max1
        max1 = D_temp1(i,i);
        maxind1 = i;
    end
end
eigvect1 = V(:,maxind1);

D_temp1(maxind1,maxind1) = 0;
max2 = D_temp1(1,1);
maxind2 = 1;
for i=1:length(D_temp1)
    if D_temp1(i,i) > max2
        max2 = D_temp1(i,i);
        maxind2 = i;
    end
end
eigvect2 = V(:,maxind2);

D_temp1(maxind2,maxind2) = 0;
max3 = D_temp1(1,1);
maxind3 = 1;
for i=1:length(D_temp1)
    if D_temp1(i,i) > max3
        max3 = D_temp1(i,i);
        maxind3 = i;
    end
end
eigvect3 = V(:,maxind3);

%3-D
eigvect_3d = [eigvect1 eigvect2 eigvect3];

w = eigvect_3d;
% Euclidean classifier
y_m1 = (w')*(m_tr1)';
y_m2 = (w')*(m_tr2)';
y_m3 = (w')*(m_tr3)';

correct1 = 0;
for i=1:length(Test1)
    y_Test1 = (w')*(Test1(i,:))';
    d1 = sqrt(sum((y_Test1-y_m1).^2));
    d2 = sqrt(sum((y_Test1-y_m2).^2));
    d3 = sqrt(sum((y_Test1-y_m3).^2));
    if d1 < d2 && d1 < d3
        correct1 = correct1+1;
    end
end
correct2 = 0;
for i=1:length(Test2)
    y_Test2 = (w')*(Test2(i,:))';
    d1 = sqrt(sum((y_Test2-y_m1).^2));
    d2 = sqrt(sum((y_Test2-y_m2).^2));
    d3 = sqrt(sum((y_Test2-y_m3).^2));
    if d2 < d1 && d2 < d3
        correct2 = correct2+1;
    end
end
correct3 = 0;
for i=1:length(Test3)
    y_Test3 = (w')*(Test3(i,:))';
    d1 = sqrt(sum((y_Test3-y_m1).^2));
    d2 = sqrt(sum((y_Test3-y_m2).^2));
    d3 = sqrt(sum((y_Test3-y_m3).^2));
    if d3 < d1 && d3 < d2
        correct3 = correct3+1;
    end
end
accuracy_fld_euclidean_4d_3d = (correct1+correct2+correct3)/(3*length(Test1));

%Bayesian Classifier

Tr1_proj  = (w')*(Tr1)';
m1_proj   = mean(Tr1_proj');
cov1_proj = inv(cov(Tr1_proj'));

Tr2_proj  = (w')*(Tr2)';
m2_proj   = mean(Tr2_proj');
cov2_proj = inv(cov(Tr2_proj'));

Tr3_proj  = (w')*(Tr3)';  
m3_proj   = mean(Tr3_proj');
cov3_proj = inv(cov(Tr3_proj'));

correct1 = 0;
for i=1:length(Test1)
    y_Test1 = (w')*(Test1(i,:))';
    d1 = (y_Test1-m1_proj')'*cov1_proj*(y_Test1-m1_proj');
    d2 = (y_Test1-m2_proj')'*cov2_proj*(y_Test1-m2_proj');
    d3 = (y_Test1-m3_proj')'*cov3_proj*(y_Test1-m3_proj');
    if d1 < d2 && d1 < d3
        correct1 = correct1+1;
    end
end
correct2 = 0;
for i=1:length(Test2)
    y_Test2 = (w')*(Test2(i,:))';
    d1 = (y_Test2-m1_proj')'*cov1_proj*(y_Test2-m1_proj');
    d2 = (y_Test2-m2_proj')'*cov2_proj*(y_Test2-m2_proj');
    d3 = (y_Test2-m3_proj')'*cov3_proj*(y_Test2-m3_proj');
    if d2 < d1 && d2 < d3
        correct2 = correct2+1;
    end
end
correct3 = 0;
for i=1:length(Test3)
    y_Test3 = (w')*(Test3(i,:))';
    d1 = (y_Test3-m1_proj')'*cov1_proj*(y_Test3-m1_proj');
    d2 = (y_Test3-m2_proj')'*cov2_proj*(y_Test3-m2_proj');
    d3 = (y_Test3-m3_proj')'*cov3_proj*(y_Test3-m3_proj');
    if d3 < d1 && d3 < d2
        correct3 = correct3+1;
    end
end
accuracy_fld_bayesian_4d_3d = (correct1+correct2+correct3)/(3*length(Test1));

%% 2-D
eigvect_2d = [eigvect1 eigvect2];

w = eigvect_2d;
%Euclidean classifier
y_m1 = (w')*(m_tr1)';
y_m2 = (w')*(m_tr2)';
y_m3 = (w')*(m_tr3)';

correct1 = 0;
for i=1:length(Test1)
    y_Test1 = (w')*(Test1(i,:))';
    d1 = sqrt(sum((y_Test1-y_m1).^2));
    d2 = sqrt(sum((y_Test1-y_m2).^2));
    d3 = sqrt(sum((y_Test1-y_m3).^2));
    if d1 < d2 && d1 < d3
        correct1 = correct1+1;
    end
end
correct2 = 0;
for i=1:length(Test2)
    y_Test2 = (w')*(Test2(i,:))';
    d1 = sqrt(sum((y_Test2-y_m1).^2));
    d2 = sqrt(sum((y_Test2-y_m2).^2));
    d3 = sqrt(sum((y_Test2-y_m3).^2));
    if d2 < d1 && d2 < d3
        correct2 = correct2+1;
    end
end
correct3 = 0;
for i=1:length(Test3)
    y_Test3 = (w')*(Test3(i,:))';
    d1 = sqrt(sum((y_Test3-y_m1).^2));
    d2 = sqrt(sum((y_Test3-y_m2).^2));
    d3 = sqrt(sum((y_Test3-y_m3).^2));
    if d3 < d1 && d3 < d2
        correct3 = correct3+1;
    end
end
accuracy_fld_euclidean_4d_2d = (correct1+correct2+correct3)/(3*length(Test1));

%Bayesian Classifier

Tr1_proj  = (w')*(Tr1)';
m1_proj   = mean(Tr1_proj');
cov1_proj = inv(cov(Tr1_proj'));

Tr2_proj  = (w')*(Tr2)';
m2_proj   = mean(Tr2_proj');
cov2_proj = inv(cov(Tr2_proj'));

Tr3_proj  = (w')*(Tr3)';  
m3_proj   = mean(Tr3_proj');
cov3_proj = inv(cov(Tr3_proj'));

correct1 = 0;
for i=1:length(Test1)
    y_Test1 = (w')*(Test1(i,:))';
    d1 = (y_Test1-m1_proj')'*cov1_proj*(y_Test1-m1_proj');
    d2 = (y_Test1-m2_proj')'*cov2_proj*(y_Test1-m2_proj');
    d3 = (y_Test1-m3_proj')'*cov3_proj*(y_Test1-m3_proj');
    if d1 < d2 && d1 < d3
        correct1 = correct1+1;
    end
end
correct2 = 0;
for i=1:length(Test2)
    y_Test2 = (w')*(Test2(i,:))';
    d1 = (y_Test2-m1_proj')'*(cov1_proj)*(y_Test2-m1_proj');
    d2 = (y_Test2-m2_proj')'*cov2_proj*(y_Test2-m2_proj');
    d3 = (y_Test2-m3_proj')'*cov3_proj*(y_Test2-m3_proj');
    if d2 < d1 && d2 < d3
        correct2 = correct2+1;
    end
end
correct3 = 0;
for i=1:length(Test3)
    y_Test3 = (w')*(Test3(i,:))';
    d1 = (y_Test3-m1_proj')'*cov1_proj*(y_Test3-m1_proj');
    d2 = (y_Test3-m2_proj')'*cov2_proj*(y_Test3-m2_proj');
    d3 = (y_Test3-m3_proj')'*cov3_proj*(y_Test3-m3_proj');
    if d3 < d1 && d3 < d2
        correct3 = correct3+1;
    end
end
accuracy_fld_bayesian_4d_2d = (correct1+correct2+correct3)/(3*length(Test1));

%% 1-D

eigvect_1d = [eigvect1];
w = eigvect_1d;

%Euclidean classifier
y_m1 = (w')*(m_tr1)';
y_m2 = (w')*(m_tr2)';
y_m3 = (w')*(m_tr3)';

correct1 = 0;
for i=1:length(Test1)
    y_Test1 = (w')*(Test1(i,:))';
    d1 = sqrt(sum((y_Test1-y_m1).^2));
    d2 = sqrt(sum((y_Test1-y_m2).^2));
    d3 = sqrt(sum((y_Test1-y_m3).^2));
    if d1 < d2 && d1 < d3
        correct1 = correct1+1;
    end
end
correct2 = 0;
for i=1:length(Test2)
    y_Test2 = (w')*(Test2(i,:))';
    d1 = sqrt(sum((y_Test2-y_m1).^2));
    d2 = sqrt(sum((y_Test2-y_m2).^2));
    d3 = sqrt(sum((y_Test2-y_m3).^2));
    if d2 < d1 && d2 < d3
        correct2 = correct2+1;
    end
end
correct3 = 0;
for i=1:length(Test3)
    y_Test3 = (w')*(Test3(i,:))';
    d1 = sqrt(sum((y_Test3-y_m1).^2));
    d2 = sqrt(sum((y_Test3-y_m2).^2));
    d3 = sqrt(sum((y_Test3-y_m3).^2));
    if d3 < d1 && d3 < d2
        correct3 = correct3+1;
    end
end
accuracy_fld_euclidean_4d_1d = (correct1+correct2+correct3)/(3*length(Test1));

%Bayesian Classifier

Tr1_proj  = (w')*(Tr1)';
m1_proj   = mean(Tr1_proj');
cov1_proj = inv(cov(Tr1_proj'));

Tr2_proj  = (w')*(Tr2)';
m2_proj   = mean(Tr2_proj');
cov2_proj = inv(cov(Tr2_proj'));

Tr3_proj  = (w')*(Tr3)';  
m3_proj   = mean(Tr3_proj');
cov3_proj = inv(cov(Tr3_proj'));

correct1 = 0;
for i=1:length(Test1)
    y_Test1 = (w')*(Test1(i,:))';
    d1 = (y_Test1-m1_proj')'*cov1_proj*(y_Test1-m1_proj');
    d2 = (y_Test1-m2_proj')'*cov2_proj*(y_Test1-m2_proj');
    d3 = (y_Test1-m3_proj')'*cov3_proj*(y_Test1-m3_proj');
    if d1 < d2 && d1 < d3
        correct1 = correct1+1;
    end
end
correct2 = 0;
for i=1:length(Test2)
    y_Test2 = (w')*(Test2(i,:))';
    d1 = (y_Test2-m1_proj')'*cov1_proj*(y_Test2-m1_proj');
    d2 = (y_Test2-m2_proj')'*cov2_proj*(y_Test2-m2_proj');
    d3 = (y_Test2-m3_proj')'*cov3_proj*(y_Test2-m3_proj');
    if d2 < d1 && d2 < d3
        correct2 = correct2+1;
    end
end
correct3 = 0;
for i=1:length(Test3)
    y_Test3 = (w')*(Test3(i,:))';
    d1 = (y_Test3-m1_proj')'*cov1_proj*(y_Test3-m1_proj');
    d2 = (y_Test3-m2_proj')'*cov2_proj*(y_Test3-m2_proj');
    d3 = (y_Test3-m3_proj')'*cov3_proj*(y_Test3-m3_proj');
    if d3 < d1 && d3 < d2
        correct3 = correct3+1;
    end
end
accuracy_fld_bayesian_4d_1d = (correct1+correct2+correct3)/(3*length(Test1));

acc11 = ['Classification Accuracy using FLD 4D to 3D using Eucidean classifier is:',num2str(accuracy_fld_euclidean_4d_3d)];
disp(acc11);
% acc21 = ['Classification Accuracy using FLD 4D to 2D using Eucidean classifier is:',num2str(accuracy_fld_euclidean_4d_2d)];
% disp(acc21);
% acc31 = ['Classification Accuracy using FLD 4D to 1D using Eucidean classifier is:',num2str(accuracy_fld_euclidean_4d_1d)];
% disp(acc31);
% acc12 = ['Classification Accuracy using FLD 4D to 3D using Bayesian classifier is:',num2str(accuracy_fld_bayesian_4d_3d)];
% disp(acc12);
% acc22 = ['Classification Accuracy using FLD 4D to 2D using Bayesian classifier is:',num2str(accuracy_fld_bayesian_4d_2d)];
% disp(acc22);
% acc32 = ['Classification Accuracy using FLD 4D to 1D using Bayesian classifier is:',num2str(accuracy_fld_bayesian_4d_1d)];
% disp(acc32);



