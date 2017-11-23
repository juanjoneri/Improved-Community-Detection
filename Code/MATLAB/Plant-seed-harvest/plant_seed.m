function P = plant_seed(C,nb_cluster,nb_sd)
n =length(C);
P = zeros(n,nb_cluster);
for i = 1:nb_cluster
    
    pos=find(C == i);
    rand_perm = randperm(length(pos));
    ext = pos(rand_perm);
    idx_num = ext(1:nb_sd);
    P(idx_num,i) = 1;
end
end