close all;
clear all;
clc;

% s=rng;
m1  = 0.25;
sd1 = sqrt(0.2);
G1 = normrnd(m1,sd1,[100,1]);
m2  = 0.7;
sd2 = sqrt(0.25);
G2 = normrnd(m2,sd2,[100,1]);

G = [G1;G2];
l = length(G);

delta = 0.1;
m = 1/delta;
n(1:m) = 0; 

Gnew=[];
for i=1:l
    if G(i)>=0 & G(i)<=1
        Gnew =[ Gnew G(i)];
        id = ceil(G(i)/delta);
        n(id) = n(id) + 1;
    end
end

    
N = sum(n);

for j=1:m
    p(j)= n(j)/(N*delta);
end

figure;
% subplot(221),
% histogram(Gnew,[0:delta:1]);
subplot(411),
myHist = histogram(Gnew,[0:delta:1]);
title('Histogram for delta=0.1 for 100 datapoints');
hold on;
myHist.Normalization = 'pdf'

% rng(s);
delta1 = 0.2;
m1 = 1/delta1;
n1(1:m1) = 0; 

Gnew1=[];
for i=1:l
    if G(i)>=0 & G(i)<=1
        Gnew1 =[ Gnew1 G(i)];
        id1 = ceil(G(i)/delta1);
        n1(id1) = n1(id1) + 1;
    end
end   
N1 = sum(n1);
for j=1:m1
    p1(j)= n1(j)/(N1*delta1);
end

subplot(412),
myHist = histogram(Gnew1,[0:delta1:1]);
title('Histogram for delta=0.2 for 100 datapoints');
hold on;
myHist.Normalization = 'pdf'

delta2 = 0.05;
m2 = 1/delta2;
n2(1:m2) = 0; 

Gnew2=[];
for i=1:l
    if G(i)>=0 & G(i)<=1
        Gnew2 =[ Gnew2 G(i)];
        id2 = ceil(G(i)/delta2);
        n2(id2) = n2(id2) + 1;
    end
end   
N2 = sum(n2);
for j=1:m2
    p2(j)= n2(j)/(N2*delta2);
end

subplot(413),
myHist = histogram(Gnew2,[0:delta2:1]);
title('Histogram for delta=0.05 for 100 datapoints');
hold on;
myHist.Normalization = 'pdf'

delta3 = 0.025;
m3 = 1/delta3;
n3(1:m3) = 0; 

Gnew3=[];
for i=1:l
    if G(i)>=0 & G(i)<=1
        Gnew3 =[ Gnew3 G(i)];
        id3 = ceil(G(i)/delta3);
        n3(id3) = n3(id3) + 1;
    end
end   
N3 = sum(n3);
for j=1:m3
    p3(j)= n3(j)/(N3*delta3);
end

subplot(414),
myHist = histogram(Gnew3,[0:delta3:1]);
title('Histogram for delta=0.025 for 100 datapoints');
hold on;
myHist.Normalization = 'pdf'


figure ;
subplot(211),
[p1,x1]= hist(G1);
plot(x1,p1/sum(p1));4
xlim([0 1]);
hold on 
[p2,x2]= hist(G2);
plot(x2,p2/sum(p2));
xlim([0 1]);
title('pdf of individual gaussian distributions');

subplot(212),
[p,x]=hist(Gnew);
plot(x,p/sum(p));
xlim([0 1]);
title('pdf of concatenated gaussian distributions');


