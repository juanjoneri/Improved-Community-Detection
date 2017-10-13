function [pur] = RW_20NEWS_more_random_sd( A,C, diffStep )

nb_cluster=20;
nb_sd=100;
n=size(A,1);
v = sum(A,2);
v = 1./v;
Dinv = spdiags(v,0,n,n);
M = A * Dinv;
P = zeros(n,nb_cluster);

for i=1:nb_cluster
    for j=1:nb_sd
    start_pos=(i-1)*1000+1;
    end_pos = start_pos+1000;
    idx_num=randi([start_pos, end_pos-1]);
    idx_num=min(n,idx_num);
    P(idx_num, i)=1;
    end
end



for i=1:diffStep
    P = M*P;
end

[~,Cnew] = max(P,[],2);
pur=purity(Cnew, C , nb_cluster);


end
