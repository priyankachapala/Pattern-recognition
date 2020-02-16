%FLD

clear all;
close all;
clc;
%% Initialisation and generation of data points

dim = 2;
n1 = 100;
m1 = [0.25 0.3];
cov1 = [0.4 0.1;0.1 0.4];
rng('default'); 
G1 = mvnrnd(m1,cov1,n1);
n2 = 100;
m2 = [0.7 0.75];
cov2 = [0.3 0.1;0.1 0.3];
rng('default'); 
G2 = mvnrnd(m2,cov2,n2);
%% Separating training and testing data points

train_size1 = 0.8; 
Tr1 = G1(1:train_size1*n1,:);
Test1 = G1((train_size1*n1)+1:n1,:);
train_size2 = 0.8;
Tr2 = G2(1:train_size2*n2,:);
Test2 = G2((train_size2*n2)+1:n2,:);
%% Finding the mean and covariance of the trained points 

m_tr1 = mean(Tr1);
cov_tr1 = cov(Tr1);
m_tr2 = mean(Tr2);
cov_tr2 = cov(Tr2);
%% Finding between class covariance matrix

SB = (m_tr2-m_tr1)'*(m_tr2-m_tr1);
%% Finding within class covariance matrix
SW = cov_tr1+cov_tr2;
%% FLD

M = inv(SW)*SB;

[V, D] =eig(M);
max = D(1,1);
maxind = 1;
for i=1:length(D)
    if D(i,i)>max
        max = D(i,i);
        maxind = i;
    end
end
eigvect = V(:,maxind);

w = eigvect;
y_m1 = (w')*(m_tr1)';
y_m2 = (w')*(m_tr2)';

if y_m1>y_m2
    class1 = 1;
    class2 = 2;
else
    class1 = 2;
    class2 = 1;
end
    
Threshold = ((y_m1 + y_m2)/2)-0.15;
correct1 = 0;
for i=1:length(Test1)
    y_Test1 = (w')*(Test1(i,:))';
    if y_Test1 > Threshold
        classi = class1;
        if classi == 1
            correct1 = correct1 +1;
        end
    else 
        classi = class2;
        if classi == 1
            correct1 = correct1 +1;
        end
    end
end
correct2 = 0;
for i=1:length(Test2)
    y_Test2 = (w')*(Test2(i,:))';
    if y_Test2 > Threshold
        classi = class1;
        if classi == 2
            correct2 = correct2 +1;
        end
    else 
        classi = class2;
        if classi == 2
            correct2 = correct2 +1;
        end
    end
end
accuracy_fld_2d_1d = (correct1+correct2)/(length(Test1)+length(Test2));

acc = ['Accuracy using FLD 2D to 1D and threshold as slightly less than middle point is:',num2str(accuracy_fld_2d_1d)];
disp(acc);
