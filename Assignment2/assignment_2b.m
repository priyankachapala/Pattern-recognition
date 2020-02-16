close all
clear all
clc

m1 = [0.25 0.3];
cov1 = [0.2 0.2;0.2 0.3];
G1 = mvnrnd(m1,cov1,100);
m2 = [0.7 0.75];
cov2 = [0.25 0.3;0.3 0.4];
G2 = mvnrnd(m2,cov2,100);

G = [G1 ;G2];
l= length(G);

D = pdist(G);
M = squareform(D);
h=0.1;
K(1:l) =0;
p(1:l,1:l)=0;
for i=1:l
    for j= 1:l
        if (M(i,j)/h) <= 0.5
            K(i) = K(i)+1;
        end
         p(i,j) = (1/(l*h*h))*K(i); 
    end
  
end

surf(G(:,1),G(:,2),p);

