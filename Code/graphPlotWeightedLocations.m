%% graphPlotWeightedLocations

% a, b represent the coordinates of the nodes
% p represents a vector with weights for painting

function p = graphPlotWeightedLocations( a, b, p )
    p = scatter(a, b, 30, p, 'filled');
    
    % Uncomment to store the image
    % print(f, 'MySavedPlot','-dpng')
end