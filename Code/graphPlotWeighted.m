%% graphPlotWeighted

% Makes a graph out of an adjacency matrix and plots it.
% w is a vector with weights on the nodes of A to apply colors

function p = graphPlotWeighted( A, w )
    f = figure
    p = plot(graph(A));
    p.NodeCData = w;
    colorbar
    
    % Uncomment to store the image
    print(f, 'MySavedPlot','-dpng')
end