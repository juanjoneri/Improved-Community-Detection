%% Take a graph that has small disconnected communities and select on only the largest community

%Load a sample Matrix with multiple communities
a = [1 2 4 5 6 7 8 8]';
b = [2 3 6 6 7 8 9 10]';
W = adjacency(graph(a, b));
dim = size(W);
dim = dim(1);
% Plot the result
graphPlot(W);


deleted = 0;  % Keep track of deleted nodes to build d
maxi = 1;     % Find node with highest degree (assume its in the main cluster)
max = 0;      % Find the degree of that node.
d = [];       % Store the degree of each node
for i=1:dim
    % Count the number of connections each node has
    s = sum(W(i-deleted,:));
    if s == 0
        % Eliminate nods with no connections
        W(i-deleted,:) = [];
        W(:,i-deleted) = [];
        deleted = deleted + 1;
    else
        d(i-deleted) = s;
    end
    if s > max
        maxi = i - deleted; %initialization node
        max = s;
    end
end

dim = size(W);       % We have a smaller graph due to some eliminations in 
dim = dim(1);        % the previous step


% Apply the random walk argorithm to the highest degree node
% to find nodes connected to it
Dh = diag(d.^(-1'));
p = zeros(dim, 1);
p(maxi) = 1;
pfinal = zeros(dim, 1);
pfinal(maxi) = 1;
for i = 1:300
    p = (Dh*W)*pfinal;
    pfinal(find(p)) = 1;
end

graphPlotWeighted(W, p);