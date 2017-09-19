function  F  = greedy( F, W, alpha, diffStep )
    
    [n,nc] = size(F);
    
    for i = 1:1000

        H = pageRank( F, W, alpha, diffStep );
        [~, maxPos] = max(H');
        F = zeros(n, nc);
        indices = sub2ind(size(F), 1:n, maxPos);
        F(indices) = 1;
       
    end
    p = clusterPlot(F, W);
end
