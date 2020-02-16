%4D data 

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

Tr = [];
n = length(Tr1);
for i= 1:n
    Tr = [Tr Tr1(i,:)' Tr2(i,:)' Tr3(i,:)'];
end

%% forward and backward pass

n_ip = 4;
n_hu = 3;
n_op = 3;

w1 = rand(n_ip,n_hu);
w2 = rand(n_hu,n_op);

for it = 1:100
for i = 1:length(Tr)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Tr(:,i));
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    if mod(i,3) == 1
        targ = [1 0 0];
    elseif mod(i,3) == 2
        targ = [0 1 0];
    elseif mod(i,3) == 0
        targ = [0 0 1];
    end
       
    delta = (out_o - targ).*out_o.*(1-out_o);
    dE_dw2 = out_h'*delta;
    w2 = w2 - dE_dw2;
    delta_h = (delta * w2').*out_h.*(1-out_h);
    dE_dw1 = Tr(:,i)*delta_h;
    w1 = w1 - dE_dw1;   
end
end

%% Training accuracy computation 

train_correct1=0;
for p = 1:length(Tr1)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Tr1(p,:)');
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o1 > out_o2 && out_o1 > out_o3
        train_correct1 = train_correct1 +1;
    end
end

train_correct2=0;
for p = 1:length(Tr2)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Tr2(p,:)');
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o2 > out_o1 && out_o2 > out_o3
        train_correct2 = train_correct2 +1;
    end
end

train_correct3=0;
for p = 1:length(Tr3)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Tr3(p,:)');
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o3 > out_o1 && out_o3 > out_o2
        train_correct3 = train_correct3 +1;
    end
end

train_accuracy=(train_correct1+train_correct2+train_correct3)/240;
%% Testing accuracy computation 

test_correct1=0;
for p = 1:length(Test1)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Test1(p,:)');
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o1 > out_o2 && out_o1 > out_o3
        test_correct1 = test_correct1 +1;
    end
end

test_correct2=0;
for p = 1:length(Test2)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Test2(p,:)');
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o2 > out_o1 && out_o2 > out_o3
        test_correct2 = test_correct2 +1;
    end
end

test_correct3=0;
for p = 1:length(Test3)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Test3(p,:)');
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o3 > out_o1 && out_o3 > out_o2
        test_correct3 = test_correct3 +1;
    end
end

test_accuracy=(test_correct1+test_correct2+test_correct3)/60;
%% displaying

disp('4D inputs fed to MLP');
no_hu = ['No.of hidden units:',num2str(n_hu)];
disp(no_hu);

a_train=['Training Accuracy: ',num2str(train_accuracy)];
disp(a_train)
c1_train=['Number of correctly classified train points in class 1: ',num2str(train_correct1)];
disp(c1_train)
c2_train=['Number of correct classified train points in class 2: ',num2str(train_correct2)];
disp(c2_train)
c3_train=['Number of correct classified train points in class 3: ',num2str(train_correct3)];
disp(c3_train)

a_test=['Testing Accuracy: ',num2str(test_accuracy)];
disp(a_test)
c1_test=['Number of correctly classified test points in class 1: ',num2str(test_correct1)];
disp(c1_test)
c2_test=['Number of correct classified test points in class 2: ',num2str(test_correct2)];
disp(c2_test)
c3_test=['Number of correct classified test points in class 3: ',num2str(test_correct3)];
disp(c3_test)
 
%% FLD reducing to 3-D
% Finding the mean and covariance of the trained points 

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

Tr_proj = w'*Tr;
Tr1_proj = w'*Tr1';
Tr2_proj = w'*Tr2';
Tr3_proj = w'*Tr3';
Test1_proj = Test1*w;
Test2_proj = Test2*w;
Test3_proj = Test3*w;

%% forward and backward pass

n_ip = 3;
n_hu = 3;
n_op = 3;

w1 = rand(n_ip,n_hu);
w2 = rand(n_hu,n_op);

for it = 1:100
for i = 1:length(Tr_proj)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Tr_proj(:,i));
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    if mod(i,3) == 1
        targ = [1 0 0];
    elseif mod(i,3) == 2
        targ = [0 1 0];
    elseif mod(i,3) == 0
        targ = [0 0 1];
    end
       
    delta = (out_o - targ).*out_o.*(1-out_o);
    dE_dw2 = out_h'*delta;
    w2 = w2 - dE_dw2;
    delta_h = (delta * w2').*out_h.*(1-out_h);
    dE_dw1 = Tr_proj(:,i)*delta_h;
    w1 = w1 - dE_dw1;   
end
end

%% Training accuracy computation 

train_correct1=0;
for p = 1:length(Tr1_proj)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Tr1_proj(:,p));
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o1 > out_o2 && out_o1 > out_o3
        train_correct1 = train_correct1 +1;
    end
end

train_correct2=0;
for p = 1:length(Tr2_proj)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Tr2_proj(:,p));
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o2 > out_o1 && out_o2 > out_o3
        train_correct2 = train_correct2 +1;
    end
end

train_correct3=0;
for p = 1:length(Tr3_proj)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Tr3_proj(:,p));
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o3 > out_o1 && out_o3 > out_o2
        train_correct3 = train_correct3 +1;
    end
end

train_accuracy=(train_correct1+train_correct2+train_correct3)/240;
%% Testing accuracy computation 

test_correct1=0;
for p = 1:length(Test1_proj)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Test1_proj(p,:)');
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o1 > out_o2 && out_o1 > out_o3
        test_correct1 = test_correct1 +1;
    end
end

test_correct2=0;
for p = 1:length(Test2_proj)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Test2_proj(p,:)');
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o2 > out_o1 && out_o2 > out_o3
        test_correct2 = test_correct2 +1;
    end
end

test_correct3=0;
for p = 1:length(Test3_proj)
    for j = 1:n_hu
        net_h(j) = sum(w1(:,j)'*Test3_proj(p,:)');
    end
    out_h = sigmf(net_h,[1 0]);
    for k = 1:n_op
        net_o(k) = sum(w2(:,k)'*out_h');
    end
    out_o = sigmf(net_o,[1 0]);
    
    out_o1 = out_o(1);
    out_o2 = out_o(2);
    out_o3 = out_o(3);
    if out_o3 > out_o1 && out_o3 > out_o2
        test_correct3 = test_correct3 +1;
    end
end

test_accuracy=(test_correct1+test_correct2+test_correct3)/60;
%% displaying

disp('4D reduced to 3D with FLD inputs fed to MLP');
no_hu = ['No.of hidden units:',num2str(n_hu)];
disp(no_hu);

a_train=['Training Accuracy: ',num2str(train_accuracy)];
disp(a_train)
c1_train=['Number of correctly classified train points in class 1: ',num2str(train_correct1)];
disp(c1_train)
c2_train=['Number of correct classified train points in class 2: ',num2str(train_correct2)];
disp(c2_train)
c3_train=['Number of correct classified train points in class 3: ',num2str(train_correct3)];
disp(c3_train)

a_test=['Testing Accuracy: ',num2str(test_accuracy)];
disp(a_test)
c1_test=['Number of correctly classified test points in class 1: ',num2str(test_correct1)];
disp(c1_test)
c2_test=['Number of correct classified test points in class 2: ',num2str(test_correct2)];
disp(c2_test)
c3_test=['Number of correct classified test points in class 3: ',num2str(test_correct3)];
disp(c3_test)


