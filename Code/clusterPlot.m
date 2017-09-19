%% clusterPlot

% Makes a graph out of an adjacency matrix and plots it.
% Uses the values of the F matrix to paint the nodes in W to each of the
% clusters

function p = clusterPlot( F, W )
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w'];
    [rows, cols] = size(F);

    p = graphPlot(W);
    
    for cluster = 1:cols
        positions = find(F(:, cluster));
        highlight(p, positions,'NodeColor',colors(cluster));
    end
end