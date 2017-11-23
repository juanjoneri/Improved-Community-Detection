%% graphPlot

% Makes a graph out of an adjacency matrix and plots it.

function p = graphPlot( A )
    f = figure;
    p = plot(graph(A));
    
    % Uncomment this line to save the output to a file
    % print(f, 'MySavedPlot','-dpng') 
end