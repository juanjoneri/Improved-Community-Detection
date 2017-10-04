clear;
W = [
     0     1     1     0     0     0     0     0     0     0
     1     0     1     1     0     0     0     0     0     0
     1     1     0     0     0     0     0     0     0     0
     0     1     0     0     1     0     1     0     0     0
     0     0     0     1     0     1     1     0     0     0
     0     0     0     0     1     0     1     0     0     0
     0     0     0     1     1     1     0     0     1     0
     0     0     0     0     0     0     0     0     1     1
     0     0     0     0     0     0     1     1     0     1
     0     0     0     0     0     0     0     1     1     0
     ];
 
dim = size(W);
dim = dim(1);

graphPlot(W, linspace(1, dim, dim));

%Degree of the nodes
for i=1:dim
    d(i) = sum(W(i,:));
end
D = diag(d');
Dh = D^(-1/2);

%Original position of random walker
p = ones(dim,1)./dim;

for i = 1:150
    p = (Dh*W*Dh)*p;
end
p'
figure;
graphPlot(W, p);
