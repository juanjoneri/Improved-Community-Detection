function [ percentage] = purity(C_computed, C_grndtruth , R)

N =length(C_grndtruth);
nb_of_dominant_points_in_class = zeros(R,1);

for k = 1:R

   v = C_grndtruth( C_computed == k  ); 
   [~,nb_of_dominant_points_in_class(k)] = mode(v);

end
percentage=(sum(nb_of_dominant_points_in_class)/N);
end