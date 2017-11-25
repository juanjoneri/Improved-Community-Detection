import networkx as nx
import matplotlib.pyplot as plt

'''
# Description
Use this module for plotting and saving plots of clusters

# More information
https://networkx.github.io/documentation/stable/tutorial.html#nodes
https://networkx.github.io/documentation/networkx-1.9/examples/drawing/weighted_graph.html
'''

def plot_G(G, coordinates=None, classes=[], cont=False):
    '''
    # Inputs
    G: an nx graph to be plotted
    coordinates: a map that specifies the coordinates of each node in G
    classes: a list that with the classes of the nodes, to give each class a unique color, or a `continous` set of values for shade panting
    '''
    colors = ['w']
    # if any(classes):
    #     # colors represent descrete classes
    #     if not cont:
    #         colors_list = ['r', 'b', 'g', 'y', 'w']
    #         colors = list(map(lambda i: colors_list[int(i)], classes))
    #     # colors represent a continuous value
    #     else:
    #         colors = classes
    colors = classes
    fig = plt.figure()
    fig.add_subplot(1,1,1)

    # title = 'Partition' if classes else 'Graph'
    # plt.title(title)
    #        x0, y0, w, h
    plt.axes([0, 0, 1, 1])
    nx.draw(G, coordinates, with_labels=True, node_color=colors)
    plt.show()

def save_G(G, file_name, plot_title='Graph'):
    '''
    # Inputs
    file_name: the name of the image to hold the plot
    '''
    fig = plt.figure()
    plt.title(plot_title)
    fig.add_subplot(1,1,1)
    plt.axes([0, 0, 1, 1])
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.savefig("{}.png".format(file_name))

if __name__ == '__main__':
    G = nx.Graph()
    G.add_node(1)
    G.add_nodes_from([2, 3, 4, 5])
    G.add_edge(1, 2)
    G.add_edges_from([(3, 4), (4, 5), (3, 5)])
    plot_G(G)
