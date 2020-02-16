clear all;
close all;
clc;
%% Initialisation and generation of data points

n1 = 1000;
m1 = [0.25 0.3];
cov1 = [0.2 0.25;0.25 0.4];
G1 = mvnrnd(m1,cov1,n1);
n2 = 1000;
m2 = [0.5 0.6];
cov2 = [0.3 0.1;0.1 0.4];
G2 = mvnrnd(m2,cov2,n2);
n3 = 1000;
m3 = [0.7 0.75];
cov3 = [0.3 0.1;0.1 0.4];
G3 = mvnrnd(m2,cov2,n2);
%% Separating training and testing data points

train_size1 = 0.8; 
Tr1 = G1(1:train_size1*n1,:);
Test1 = G1((train_size1*n1)+1:n1,:);
train_size2 = 0.8;
Tr2 = G2(1:train_size2*n2,:);
Test2 = G2((train_size2*n2)+1:n2,:);
train_size3 = 0.8;
Tr3 = G3(1:train_size3*n3,:);
Test3 = G3((train_size2*n3)+1:n3,:);
%% Finding the mean and covariance of the trained points and giving priors

m_tr1 = mean(Tr1);
cov_tr1 = cov(Tr1);
m_tr2 = mean(Tr2);
cov_tr2 = cov(Tr2);
m_tr3 = mean(Tr3);
cov_tr3 = cov(Tr3);

pc1 = 1/3;
pc2 = 1/3;
pc3 = 1/3;
%% Training 

train_correct1=0;
for i =1:length(Tr1)
    disc_a = disc(Tr1(i),m_tr1,cov_tr1,pc1);
    disc_b = disc(Tr1(i),m_tr2,cov_tr2,pc2);
    disc_c = disc(Tr1(i),m_tr3,cov_tr3,pc3);
    if disc_a > disc_b && disc_a > disc_c
       train_correct1 = train_correct1+1; 
    end
end
train_correct2=0;
for i =1:length(Tr2)
    disc_a = disc(Tr2(i),m_tr1,cov_tr1,pc1);
    disc_b = disc(Tr2(i),m_tr2,cov_tr2,pc2);
    disc_c = disc(Tr2(i),m_tr3,cov_tr3,pc3);
    if disc_b > disc_a && disc_b > disc_c
       train_correct2 = train_correct2+1; 
    end
end
train_correct3=0;
for i =1:length(Tr3)
    disc_a = disc(Tr3(i),m_tr1,cov_tr1,pc1);
    disc_b = disc(Tr3(i),m_tr2,cov_tr2,pc2);
    disc_c = disc(Tr3(i),m_tr3,cov_tr3,pc3);
    if disc_c > disc_a && disc_c > disc_b
       train_correct3 = train_correct3+1; 
    end
end

train_acc = (train_correct1+train_correct2+train_correct3)/(length(Tr1)+length(Tr2)+length(Tr3));
%% Testing

test_correct1=0;
for i =1:length(Test1)
    disc_a = disc(Test1(i),m_tr1,cov_tr1,pc1);
    disc_b = disc(Test1(i),m_tr2,cov_tr2,pc2);
    disc_c = disc(Test1(i),m_tr3,cov_tr3,pc3);
    if disc_a > disc_b && disc_a > disc_c
       test_correct1 = test_correct1+1; 
    end
end
test_correct2=0;
for i =1:length(Test2)
    disc_a = disc(Test2(i),m_tr1,cov_tr1,pc1);
    disc_b = disc(Test2(i),m_tr2,cov_tr2,pc2);
    disc_c = disc(Test2(i),m_tr3,cov_tr3,pc3);
    if disc_b > disc_a && disc_b > disc_c
       test_correct2 = test_correct2+1; 
    end
end
test_correct3=0;
for i =1:length(Test3)
    disc_a = disc(Test3(i),m_tr1,cov_tr1,pc1);
    disc_b = disc(Test3(i),m_tr2,cov_tr2,pc2);
    disc_c = disc(Test3(i),m_tr3,cov_tr3,pc3);
    if disc_c > disc_a && disc_c > disc_b
       test_correct3 = test_correct3+1; 
    end
end

test_acc = (test_correct1+test_correct2+test_correct3)/(length(Test1)+length(Test2)+length(Test3));

%% Display
num1 = ['Total points from class1:',num2str(n1)];
disp(num1)
num2 = ['Total points from class2:',num2str(n2)];
disp(num2)
num3 = ['Total points from class3:',num2str(n3)];
disp(num3)

train_correct=['Number of training points correctly classified: ',num2str(train_correct1+train_correct2+train_correct3)];
disp(train_correct)
test_correct=['Number of test points correctly classified: ',num2str(test_correct1+test_correct2+test_correct3)];
disp(test_correct)

train=['Training Accuracy is: ',num2str(train_acc)];
disp(train)
test=['Testing Accuracy is: ',num2str(test_acc)];
disp(test)



