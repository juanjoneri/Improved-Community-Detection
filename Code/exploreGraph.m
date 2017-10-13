function [ p ] = exploreGraph( W, X, alpha,nb_steps )

    x_coordinates = X(:,1);
    y_coordinates = X(:,2);

    dim = size(W);
    dim = dim(1);
    d = sum(W, 1);

    Dh = diag(d.^(-1/2)');
    M = Dh*W*Dh;

    %Original position of random walker
    p = zeros(dim, 1);
    p(randi([1, dim])) = 1;
    p_initial = p;
    
    for i = 1:nb_steps
        
        p = alpha*(M*p) + (1-alpha)*p_initial;
        
        if mod(i,floor(nb_steps/20)) == 0
            graphPlotWeightedLocations(x_coordinates, y_coordinates, p);
            pause
        end
    end
    
end

