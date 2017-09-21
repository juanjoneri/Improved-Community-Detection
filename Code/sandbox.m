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
 
graphPlot(W);

%Degree of the nodes
for i=1:10
    D(i) = sum(W(i,:));
end
D = diag(D');
Dm = D^(-1);

%Original position of random walker
p = ones(10,1)./10;

graphPlot(W);
for i = 1:150
    p = W*(Dm*p);
end
p'
sum(p)
