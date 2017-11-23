function [ C ] = heat_dist_random_walk( A, X, diffStep )

n=size(A,1);
v = sum(A,1);
v = 1./v;
Dinv = diag(v);
M =  Dinv^(1/2) * A * Dinv^(1/2) ;
P = zeros(n,3);

idx1=randi([1,100]);
idx2=randi([101,200]);
idx3=randi([201,300]);
%idx1=1;
%idx2=101;
%idx3=201;

P(idx1, 1) = 1;
P(idx2, 2) = 1;
P(idx3, 3) = 1;

C=zeros(n,1);
C(idx1)=1; 
C(idx2)=1;
C(idx3)=1;
scatter(X(:,1), X(:,2), 40, C,'filled')
pause



for i=1:diffStep
    P = M*P;
    if mod(i,20) == 0
        scatter(X(:,1), X(:,2), 40,  P(:,1), 'filled');
        %scatter(X(:,1), X(:,2), 40,  P(:,2), 'filled');
    % scatter(X(:,1), X(:,2), 40,  P(:,3), 'filled');
        title(['step', num2str(i)])
        pause
    end
end

[~,C] = max(P,[],2);

scatter(X(:,1), X(:,2), 30, C,'filled');


end
