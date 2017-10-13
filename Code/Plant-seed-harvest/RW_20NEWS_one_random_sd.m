function [pur] = RW_20NEWS_one_random_sd( A,C, diffStep )

nb_cluster=20;
nb_sd=100;
n=size(A,1);
v = sum(A,2);
v = 1./v;
Dinv = spdiags(v,0,n,n);
M = A * Dinv;


P=plant_seed(C,nb_cluster,nb_sd);

for i = 1:diffStep
    P = M*P;
end

[~,Cnew] = max(P,[],2);
pur = purity(Cnew, C , nb_cluster);
end

% 
% function plant_seed(C,nb_sd)
% 
% for i=1:nb_cluster
%     
%     pos=find(C==i);
%     rand_perm=randperm(length(pos));
%     ext=pos(rand_perm);
%     idx_num=ext(1:nb_sd);
%     P(idx_num,i)=1;
% end
% 
% end