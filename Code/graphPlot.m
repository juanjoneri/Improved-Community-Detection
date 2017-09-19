%% graphPlot

% Makes a graph out of an adjacency matrix and plots it.

function p = graphPlot( A )
    p = plot(graph(A));
end