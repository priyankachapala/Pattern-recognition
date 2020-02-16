close all
clear all
clc

m1 = [0.25 0.3];
cov1 = [0.3 0.2;0.2 0.45];
G1 = mvnrnd(m1,cov1,100);
m2 = [0.7 0.75];
cov2 = [0.25 0.3;0.3 0.4];
G2 = mvnrnd(m2,cov2,100);

G = [G1 ;G2];
l= length(G);

h = 0.1;
range=0:h:1;
h_count=hist3(G,'Nbins',[size(range,2) size(range,2)]); 
hist3(G,'Nbins',[size(range,2) size(range,2)]);
title('Histogram for h=0.1');
p=h_count/(h*h*size(G,1)); 
figure
surf(range,range,p);
title('Normalised pdf for h=0.1');

h = 0.005;
range=0:h:1;
h_count=hist3(G,'Nbins',[size(range,2) size(range,2)]); 
% hist3(G,'Nbins',[size(range,2) size(range,2)]);
% title('Histogram for h=0.005');
p=h_count/(h*h*size(G,1)); 
figure
subplot(211),
surf(range,range,p);
title('Normalised pdf for h=0.005');

h = 0.06;
range=0:h:1;
h_count=hist3(G,'Nbins',[size(range,2) size(range,2)]); 
% hist3(G,'Nbins',[size(range,2) size(range,2)]);
% title('Histogram for h=0.06');
p=h_count/(h*h*size(G,1)); 
subplot(212),
surf(range,range,p);
title('Normalised pdf for h=0.06');

h = 0.2;
range=0:h:1;
h_count=hist3(G,'Nbins',[size(range,2) size(range,2)]); 
% hist3(G,'Nbins',[size(range,2) size(range,2)]);
% title('Histogram for h=0.2');
p=h_count/(h*h*size(G,1));
figure
subplot(211),
surf(range,range,p);
title('Normalised pdf for h=0.2');

h = 0.5;
range=0:h:1;
h_count=hist3(G,'Nbins',[size(range,2) size(range,2)]); 
% hist3(G,'Nbins',[size(range,2) size(range,2)]);
% title('Histogram for h=0.5');
p=h_count/(h*h*size(G,1)); 
subplot(212),
surf(range,range,p);
title('Normalised pdf for h=0.5');
