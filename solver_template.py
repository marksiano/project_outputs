import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from student_utils_sp18 import *
import networkx as nx
import numpy as np
import pdb
import priorityqueue

MAXPRIORITY = 999999
WEIGHT = None

"""
======================================================================
  Complete the following function.
======================================================================
"""

def weight(G):
    total_weight = 0.0
    for node in [node for node in G.nodes()]:
        total_weight += G[node][node]['weight']

    return total_weight

def conquer_cost(G, target_node):
    return G[target_node][target_node]['weight']

def acquire_cost(G, current_node, target_node, shortest_paths):
    shortest_path = shortest_paths[current_node][target_node]
    cost = conquer_cost(G, target_node)

    factors = np.array([shortest_path, cost])
    weights = np.array([1.0, 1.0])

    acquire_cost = np.dot(factors, weights)

    return np.float64(acquire_cost).item()

def value(G, current_node, target_node, unconquered_kingdoms, conquered_kingdoms, start_node):

    adjacent = [neighbor for neighbor in G.neighbors(target_node) if (neighbor != target_node and neighbor in unconquered_kingdoms)]
    outreach = (len([a for a in adjacent]) + 1) / len(unconquered_kingdoms)
    outreach = outreach ** 10   # What percent of remaining nodes it will conquer

    coming_from_leaf = False
    leaf_bonus = 0.0

    current_node_adjacent = [neighbor for neighbor in G.neighbors(current_node) if neighbor != current_node]
    unconquered_adjacent = list(set(current_node_adjacent).intersection(unconquered_kingdoms))
    #pdb.set_trace()

    # The node you are coming from only has one remaining unconquered adjacent node: the target
    if len(unconquered_adjacent) == 1 and unconquered_adjacent[0] == target_node:
        coming_from_leaf = True

    if coming_from_leaf and start_node != current_node:
        # You get a bonus for stopping the inevitable return to capture the leaf
        # In addition, avoid adding the leaf bonus if the leaf node is the start node, since you will have to 
        # return to the start node anyway
        leaf_bonus = G[current_node][current_node]['weight']    # The weight of the leaf
        #leaf_bonus = 0.0
    
    # Includes conquer cost for current node, since it is adjacent to itself
    surrender_value = sum([conquer_cost(G, neighbor) for neighbor in adjacent])

    factors = np.array([outreach, surrender_value, leaf_bonus])
    #print("Factors for node", target_node, ":", outreach, ",", surrender_value)
    weights = np.array([10 * weight(G), 1.0, weight(G) * 10])
    # For weights: Everything not implicitly scaled by the weight of the graph should be explicitly multiplied by it

    return np.float64(np.dot(factors, weights)).item()


def solve(list_of_kingdom_names, starting_kingdom, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_kingdom_names: An list of kingdom names such that node i of the graph corresponds to name index i in the list
        starting_kingdom: The name of the starting kingdom for the walk
        adjacency_matrix: The adjacency matrix from the input file

    Output:
        Return 2 things. The first is a list of kingdoms representing the walk, and the second is the set of kingdoms that are conquered
    """
    
    print("INPUT")
    print("-----")
    print("Adjacency matrix: ", adjacency_matrix)
    print("Kingdom names: ", list_of_kingdom_names)
    N = len(adjacency_matrix)   # number of vertices
    closed_walk = []
    conquered_kingdoms = set()

    shortest_distance = [[float("inf") for i in range(N)] for j in range(N)]

    G = nx.Graph()
    edges = []

    for i, row in enumerate(adjacency_matrix):
        for j, item in enumerate(adjacency_matrix[i]):
            weight = adjacency_matrix[i][j]
            if weight == 'x':
                continue

            edges.append((i, j, weight))
    
    G.add_weighted_edges_from(edges)

    shortest_paths = nx.floyd_warshall(G)

    vertex_sets = [None] * len(G.nodes())
    costs = []
    start_vertex = list_of_kingdom_names.index(starting_kingdom)

    #pdb.set_trace()

    for i, vertex in enumerate(G.nodes()):
        vertex_sets[i] = []
        costs.append(acquire_cost(G, start_vertex, vertex, shortest_paths))
        for adjacent_vertex in G[vertex]:
            vertex_sets[i].append(adjacent_vertex)

    print("Initial costs: ", costs)
    print()

    selected, cost = weightedsetcover(G, vertex_sets, costs, start_vertex, shortest_paths)

    print("RESULT")
    print("------")
    print(selected, " - cost: ", cost)

    return selected, conquered_kingdoms

def weightedsetcover(G, S, costs, start_node, shortest_paths):
    '''Weighted set cover greedy algorithm:
    pick the set which is the most cost-effective: min(w[s]/|s-C|),
    where C is the current covered elements set.
    The complexity of the algorithm: O(|U| * log|S|) .
    Finding the most cost-effective set is done by a priority queue.
    The operation has time complexity of O(log|S|).
    Input:
    udict - universe U, which contains the <elem, setlist>. (dict)
    S - a collection of sets. (list)
    w - corresponding weight to each set in S. (list)
    Output:
    selected: the selected set ids in order. (list)
    cost: the total cost of the selected sets.
    '''

    K = len(S)
    udict = {}
    selected = list()
    adj = [] # During the process, S will be modified. Make a copy for S.
    for index, item in enumerate(S):
        adj.append(set(item))
        for j in item:
            if j not in udict:
                udict[j] = set()
            udict[j].add(index)

    conquered_kingdoms = set()
    unconquered_kingdoms = set([i for i in range(K)])
    current_node = start_node

    print("PQ INITIALIZATION")
    print("-----------------")

    pq = priorityqueue.PriorityQueue()
    cost = 0
    coverednum = 0
    for index, node in enumerate(adj): # add all sets to the priorityqueue
        if len(node) == 0:
            pq.addtask(index, MAXPRIORITY)
            print("Added node", index, "with priority", MAXPRIORITY)
        else:
            priority = float(costs[index]) / value(G, start_node, index, unconquered_kingdoms, conquered_kingdoms, start_node)
            pq.addtask(index, priority)
            print("Added node", index, "with priority", priority, "(cost:", float(costs[index]), ", value:", value(G, start_node, index, unconquered_kingdoms, conquered_kingdoms, start_node), ")")

    print()

    while len(conquered_kingdoms) < K:
        target_node = pq.poptask() # get the most cost-effective set
        print("Conquering node: ", target_node)
        selected.append(target_node) # a: set id
        cost += costs[target_node]
        coverednum += len(adj[target_node])
        conquered_kingdoms.add(target_node)
        if target_node in unconquered_kingdoms:
            unconquered_kingdoms.remove(target_node)

        # pdb.set_trace()

        for adjacent_vertex in [node for node in adj[target_node] if node != target_node]:
            conquered_kingdoms.add(adjacent_vertex)
            if adjacent_vertex in unconquered_kingdoms:
                unconquered_kingdoms.remove(adjacent_vertex)
            #pq.addtask(adjacent_vertex, MAXPRIORITY)
            print("Adding task", adjacent_vertex, "with priority", MAXPRIORITY)
            for n in udict[adjacent_vertex]:
                if n != target_node:
                    adj[n].discard(adjacent_vertex)
                    # if len(adj[n]) == 0:
                    #     pq.addtask(n, MAXPRIORITY)

                    # else:
                    #     costs[n] = acquire_cost(G, target_node, n, shortest_paths)
                    #     #pdb.set_trace()
                    #     pq.addtask(n, costs[n] / value(G, target_node, n, unconquered_kingdoms, conquered_kingdoms))

        # Update the sets that contains the new covered elements
        # Commented out: only updates nodes that touched adjacent nodes that were removed
        # for m in adj[target_node]: # m: element
        #     for n in udict[m]:  # n: set id
        #         if n != target_node:
        #             adj[n].discard(m)
        #             if len(adj[n]) == 0:
        #                 pq.addtask(n, MAXPRIORITY)
        #             else:
        #                 pq.addtask(n, float(costs[n]) / len(adj[n]))

        # Update the acquire cost of each unacquired vertex
        for vertex in unconquered_kingdoms:
            cost = acquire_cost(G, target_node, vertex, shortest_paths)
            pq.addtask(vertex, cost / value(G, target_node, vertex, unconquered_kingdoms, conquered_kingdoms, start_node))


        adj[target_node].clear()
        pq.addtask(target_node, MAXPRIORITY)
                        
    return selected, cost


"""
======================================================================
   No need to change any code below this line
======================================================================
"""


def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)
    
    input_data = utils.read_file(input_file)
    number_of_kingdoms, list_of_kingdom_names, starting_kingdom, adjacency_matrix = data_parser(input_data)
    closed_walk, conquered_kingdoms = solve(list_of_kingdom_names, starting_kingdom, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    output_filename = utils.input_to_output(filename)
    output_file = f'{output_directory}/{output_filename}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    utils.write_data_to_file(output_file, closed_walk, ' ')
    utils.write_to_file(output_file, '\n', append=True)
    utils.write_data_to_file(output_file, conquered_kingdoms, ' ', append=True)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
