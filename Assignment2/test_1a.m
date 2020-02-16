close all;
clear all;
clc;

m1  = 0.25;
sd1 = sqrt(0.2);
G1  = (sd1*randn(100,1)) + m1;
m2  = 0.7;
sd2 = sqrt(0.25);
G2  = (sd2*randn(100,1)) + m2;

G = [G1' G2'];
l = length(G);

delta = 0.01;
m = 1/delta;
n(1:m) = 0; 

for i=1:l
    if G(i)>=0 & G(i)<=1
        id = ceil(G(i)/delta);
        n(id) = n(id) + 1;
    end
end
    
N = sum(n);

for j=1:m
    p(j)= n(j)/(N*delta);
end

figure;
hist(G,m);






