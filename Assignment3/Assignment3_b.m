clear all;
close all;
clc;
%% Initialisation and generation of data points
n1 = 50;
m1 = [0.25 0.3];
cov1 = [0.2 0.25;0.25 0.4];
G1 = mvnrnd(m1,cov1,n1);
n2 = 150;
m2 = [0.5 0.4];
cov2 = [0.3 0.1;0.1 0.4];
G2 = mvnrnd(m2,cov2,n2);
%% Separating training and testing data points

train_size1 = 0.8; 
Tr1 = G1(1:train_size1*n1,:);
Test1 = G1((train_size1*n1)+1:n1,:);
train_size2 = 0.8;
Tr2 = G2(1:train_size2*n2,:);
Test2 = G2((train_size2*n2)+1:n2,:);
%% Finding the mean and covariance of the trained points and giving priors

m_tr1 = mean(Tr1);
cov_tr1 = cov(Tr1);
m_tr2 = mean(Tr2);
cov_tr2 = cov(Tr2);

pc1 = 0.25;
pc2 = 0.75;
%% Training 

train_correct1=0;
for i =1:length(Tr1)
    disc_a = disc(Tr1(i),m_tr1,cov_tr1,pc1);
    disc_b = disc(Tr1(i),m_tr2,cov_tr2,pc2);
    if disc_a > disc_b
       train_correct1 = train_correct1+1; 
    end
end
train_correct2=0;
for i =1:length(Tr2)
    disc_a = disc(Tr2(i),m_tr1,cov_tr1,pc1);
    disc_b = disc(Tr2(i),m_tr2,cov_tr2,pc2);
    if disc_b > disc_a
       train_correct2 = train_correct2+1; 
    end
end

train_acc = (train_correct1+train_correct2)/(length(Tr1)+length(Tr2));
%% Testing

test_correct1=0;
for i =1:length(Test1)
    disc_a = disc(Test1(i),m_tr1,cov_tr1,pc1);
    disc_b = disc(Test1(i),m_tr2,cov_tr2,pc2);
    if disc_a > disc_b
       test_correct1 = test_correct1+1; 
    end
end
test_correct2=0;
for i =1:length(Test2)
    disc_a = disc(Test2(i),m_tr1,cov_tr1,pc1);
    disc_b = disc(Test2(i),m_tr2,cov_tr2,pc2);
    if disc_b > disc_a
       test_correct2 = test_correct2+1; 
    end
end

test_acc = (test_correct1+test_correct2)/(length(Test1)+length(Test2));

%% Display
num1 = ['Total points from class1:',num2str(n1)];
num2 = ['Total points from class2:',num2str(n2)];
disp(num1)
disp(num2)

train_correct=['Number of training points correctly classified: ',num2str(train_correct1+train_correct2)];
disp(train_correct)
test_correct=['Number of test points correctly classified: ',num2str(test_correct1+test_correct2)];
disp(test_correct)

train=['Training Accuracy is: ',num2str(train_acc)];
disp(train)
test=['Testing Accuracy is: ',num2str(test_acc)];
disp(test)



