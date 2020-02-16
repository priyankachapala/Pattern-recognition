close all
clear all
clc
M = input('Enter the no.of random variables ');
n = input('Enter the no.of values each random variable has ');
iter = input('Enter the  no.of iterations ');

%% Uniform Distribution

for i = 1:M
    rv_set(i,:) = rand(1,n);
end

mean_rv = [];
for j = 1:iter
    r_idx = randi(n,[1,M]); 
    for k = 1:M
     rv_values(k) = rv_set(k ,r_idx(k));
    end
    m = sum(rv_values)/M;
    mean_rv = [mean_rv m];
end
figure;
hist(mean_rv,50);
title('Histogram for mean of uniformly distributed random variables');

%% Exponential Distribution

for i = 1:M
    rv_set(i,:) = exprnd(0.5,[1 n]);
end

mean_rv = [];
for j = 1:iter
    r_idx = randi(n,[1,M]); 
    for k = 1:M
     rv_values(k) = rv_set(k ,r_idx(k));
    end
    m = sum(rv_values)/M;
    mean_rv = [mean_rv m];
end
figure;
hist(mean_rv,50);
title('Histogram for mean of exponentially distributed random variables');







    
    