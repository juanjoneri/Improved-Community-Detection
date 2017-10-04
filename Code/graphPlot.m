%% graphPlot

% Makes a graph out of an adjacency matrix and plots it.

function p = graphPlot( A, w )
    p = plot(graph(A));
    p.NodeCData = w;
end