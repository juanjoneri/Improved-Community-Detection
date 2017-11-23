import networkx as nx
import csv
import numpy as np
'''
# Description
Use this module for saving and retreiving graphs, its connections, coordinates and labels.
'''

def export_cluster(G, file_name):
    '''
    Writes the contents of a graph as an adjacency list to a file

    # Inputs:
    G: a graph to save
    file_name: the name of the file .csv to write to
    '''
    with open('{}.csv'.format(file_name), 'w') as result:
        writer = csv.writer(result, dialect='excel')
        writer.writerows(G.edges)

def export_labels(labels, coordinates, file_name):
    '''
    Writes the ground truth of the classes of the graph to filname.txt

    # Inputs:
    G: a graph to save
    file_name: the name of the file .csv to write to
    '''
    with open('{}.txt'.format(file_name), 'w') as result:
        for i in range(len(labels)):
            result.write('{}, {}, {}\n'.format(i, labels[i], coordinates[i]))

def import_cluster(file_name):
    return nx.read_edgelist(file_name, delimiter=",")

def import_labels(file_name):
    pass
