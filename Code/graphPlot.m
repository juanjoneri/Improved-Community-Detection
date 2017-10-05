%% graphPlot

% Makes a graph out of an adjacency matrix and plots it.

function p = graphPlot( A, w )
    f = figure
    p = plot(graph(A));
    p.NodeCData = w;
    colorbar
    print(f, 'MySavedPlot','-dpng')
end