%% graphPlotWeightedLocationsHeights

% a, b represent the coordinates of the nodes
% p represents a vector with weights for painting

function p = graphPlotWeightedLocationsHeights( a, b, p )
    p = scatter3(a, b, p, 30, p, 'filled');
    
    % Uncomment to store the image
    % print(f, 'MySavedPlot','-dpng')
end