import networkx as nx
import matplotlib.pyplot as plt
## https://networkx.github.io/documentation/stable/tutorial.html#nodes

def plot_G(G, coordinates, colors):
    plt.plot()
    nx.draw(G, coordinates, with_labels=True, node_color=colors)
    plt.show()

def save_G(G, name):
    plt.plot()
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.savefig("{}.png".format(name))

if __name__ == '__main__':
    G = nx.Graph()
    G.add_node(1)
    G.add_nodes_from([2, 3, 4, 5])
    G.add_edge(1, 2)
    G.add_edges_from([(3, 4), (4, 5), (3, 5)])
    plot_G(G)
