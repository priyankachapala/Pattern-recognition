%PCA reconstruction

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
rng('default'); 
G1 = mvnrnd(m1,cov1,n1);
n2 = 100;
m2 = [0.1 0.1 0.25 0.2];
cov2 = [0.25 0 0 0 ;
        0 0.1 0 0 ;
        0 0 0.375 0;
        0 0 0 0.25];
rng('default'); 
G2 = mvnrnd(m2,cov2,n2);
n3 = 100;
m3 = [1.5 1.25 1.75 2];
cov3 = [0.5 0 0 0 ;
        0 0.2 0 0 ;
        0 0 0.45 0;
        0 0 0 0.3];
rng('default'); 
G3 = mvnrnd(m3,cov3,n3);


%% Finding the mean and covariance  

m_tr1 = mean(G1);
cov_tr1 = cov(G1);
m_tr2 = mean(G2);
cov_tr2 = cov(G2);
m_tr3 = mean(G3);
cov_tr3 = cov(G3);


%% PCA
%3-D

M = cov_tr1+cov_tr2+cov_tr3;

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
%% 3-D

eigvect_3d = [eigvect1 eigvect2 eigvect3];

w = eigvect_3d;
winv=pinv(w');

y1_3d = (w')*(G1');
y2_3d = (w')*(G2');
y3_3d = (w')*(G3');

G1_recov_3d = winv*y1_3d;
G1_recov_3d = G1_recov_3d';
G2_recov_3d = winv*y2_3d;
G2_recov_3d = G2_recov_3d';
G3_recov_3d = winv*y3_3d;
G3_recov_3d = G3_recov_3d';

errorG1_3d = norm(G1 - G1_recov_3d);
errorG2_3d = norm(G2 - G2_recov_3d);
errorG3_3d = norm(G3 - G3_recov_3d);

%% 2-D

eigvect_2d = [eigvect1 eigvect2];

w = eigvect_2d;
winv=pinv(w');

y1_2d = (w')*(G1');
y2_2d = (w')*(G2');
y3_2d = (w')*(G3');

G1_recov_2d = winv*y1_2d;
G1_recov_2d = G1_recov_2d';
G2_recov_2d = winv*y2_2d;
G2_recov_2d = G2_recov_2d';
G3_recov_2d = winv*y3_2d;
G3_recov_2d = G3_recov_2d';

errorG1_2d = norm(G1 - G1_recov_2d);
errorG2_2d = norm(G2 - G2_recov_2d);
errorG3_2d = norm(G3 - G3_recov_2d);
%% 1-D
eigvect_1d = eigvect1;

w = eigvect_1d;
winv=pinv(w');

y1_1d = (w')*(G1');
y2_1d = (w')*(G2');
y3_1d = (w')*(G3');

G1_recov_1d = winv*y1_1d;
G1_recov_1d = G1_recov_1d';
G2_recov_1d = winv*y2_1d;
G2_recov_1d = G2_recov_1d';
G3_recov_1d = winv*y3_1d;
G3_recov_1d = G3_recov_1d';

errorG1_1d = norm(G1 - G1_recov_1d);
errorG2_1d = norm(G2 - G2_recov_1d);
errorG3_1d = norm(G3 - G3_recov_1d);

err_3d = ['Mean sqaure error after reconstruction from 3D for G1 is:',num2str(errorG1_3d),', G2 is:',num2str(errorG2_3d),', G3 is:',num2str(errorG3_3d)];
disp(err_3d);
err_2d = ['Mean sqaure error after reconstruction from 2D for G1 is:',num2str(errorG1_2d),', G2 is:',num2str(errorG2_2d),', G3 is:',num2str(errorG3_2d)];
disp(err_2d);
err_1d = ['Mean sqaure error after reconstruction from 1D for G1 is:',num2str(errorG1_1d),', G2 is:',num2str(errorG2_1d),', G3 is:',num2str(errorG3_1d)];
disp(err_1d);



