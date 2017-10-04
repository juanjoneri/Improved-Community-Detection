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
 
graphPlot(W, linspace(1, 10, 10));

%Degree of the nodes
for i=1:10
    d(i) = sum(W(i,:));
end
D = diag(d');
Dh = D^(-1/2);

%Original position of random walker
p = ones(10,1)./10;

for i = 1:150
    p = (Dh*W*Dh)*p;
end
p'
figure;
graphPlot(W, p);
