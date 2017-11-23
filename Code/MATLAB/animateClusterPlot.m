%% animateClusterPlot

% updates the colors of the output of a clusterPlot to the new information
% for the clusters in F


function p = animateClusterPlot( p, F, W )
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w'];
    [rows, cols] = size(F);
    
    for cluster = 1:cols
        positions = find(F(:, cluster));
        highlight(p, positions,'NodeColor',colors(cluster));
    end
    drawnow
    pause(0.001);
end