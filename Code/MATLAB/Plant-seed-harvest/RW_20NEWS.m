function [pur ] = RW_20NEWS( A,C, diffStep )
tic
nb_cluster=20;
n=size(A,1);
v = sum(A,2);
v = 1./v;
%Dinv=diag(v);
Dinv = spdiags(v,0,n,n);
M = A * Dinv;
P = zeros(n,nb_cluster);
idx_num = zeros(1,nb_cluster);


for i=1:nb_cluster
    start_pos=(i-1)*1000+1;
    end_pos = start_pos+1000;
    idx_num(i)=randi([start_pos, end_pos-1]);
    P(idx_num(i), i)=1;
end



for i=1:diffStep
    P = M*P;
end

[~,Cnew] = max(P,[],2);
pur=purity(Cnew, C , nb_cluster);
toc

end
