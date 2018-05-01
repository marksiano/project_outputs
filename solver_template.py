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
import random
import time

MAXPRIORITY = 999999
alpha = 0.3

"""
======================================================================
  Complete the following function.
======================================================================
"""

def total_cost(G, tour, kingdoms_conquered, heuristic_value, average_cost):
    global alpha
    travel_cost = 0.0
    for i in range(0, len(tour) - 1):
        travel_cost += G[tour[i]][tour[i + 1]]['weight']

    conquer_cost = 0.0
    for kingdom in kingdoms_conquered:
        conquer_cost += G[kingdom][kingdom]['weight']

    total_cost = travel_cost + conquer_cost

    if average_cost != None:
        cost_difference = average_cost - total_cost
        for i in range(0, len(kingdoms_conquered) - 1):
            if cost_difference > 0.0:
                heuristic_value[kingdoms_conquered[i]][kingdoms_conquered[i + 1]] += (alpha) * cost_difference
                alpha += 0.01
                if alpha > 1.0:
                    alpha = 1.0
                #print(alpha)

    return travel_cost + conquer_cost

def floyd_warshall(G):

    V = len(G.nodes())
    dist = [[float("inf") for x in range(V)] for y in range(V)]
    pred = [[None for x in range(V)] for y in range(V)]

    for u, v in [edge for edge in G.edges()]:
        if u == v:
            dist[u][v] = 0
            pred[u][v] = v
        else:
            dist[u][v] = G[u][v]['weight']
            dist[v][u] = G[v][u]['weight']
            pred[u][v] = v
            pred[v][u] = u

    for k in range(0, V):
        for i in range(0, V):
            for j in range(0, V):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[i][k]

                    dist[j][i] = dist[k][i] + dist[j][k]
                    pred[j][i] = pred[j][k]

    return dist, pred

def path(u, v, pred):
    if pred[u][v] == None:
        return []

    path = [u]
    while u != v:
        u = pred[u][v]
        path.append(u)

    return path

def calc_weight(G):
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
    weights = np.array([2.0, 2.0])

    acquire_cost = np.dot(factors, weights)

    return np.float64(acquire_cost).item()

def value(G, current_node, target_node, unconquered_kingdoms, conquered_kingdoms, start_node, weight, heuristic_value, randomness):

    adjacent = [neighbor for neighbor in G.neighbors(target_node) if (neighbor != target_node and neighbor in unconquered_kingdoms)]
    #outreach = (len([a for a in adjacent]) + 1) / len(unconquered_kingdoms)
    #outreach = outreach ** 10   # What percent of remaining nodes it will conquer

    #num_adjacent = len([a for a in adjacent]) + 1

    #coming_from_leaf = False
    #leaf_bonus = 0.0

    current_node_adjacent = [neighbor for neighbor in G.neighbors(current_node) if neighbor != current_node]
    unconquered_adjacent = list(set(current_node_adjacent).intersection(unconquered_kingdoms))
    #pdb.set_trace()

    # The node you are coming from only has one remaining unconquered adjacent node: the target
    # if len(unconquered_adjacent) == 1 and unconquered_adjacent[0] == target_node:
    #     coming_from_leaf = True

    # if coming_from_leaf:
    #     # You get a bonus for stopping the inevitable return to capture the leaf
    #     # In addition, avoid adding the leaf bonus if the leaf node is the start node, since you will have to 
    #     # return to the start node anyway
    #     leaf_bonus = G[current_node][current_node]['weight']    # The weight of the leaf
        #leaf_bonus = 0.0
    
    # Includes conquer cost for current node, since it is adjacent to itself
    surrender_value = sum([conquer_cost(G, neighbor) for neighbor in adjacent])

    factors = np.array([1, surrender_value, 0.0, 0.25 if randomness else 0.0, heuristic_value[current_node][target_node]])
    #print("Factors for node", target_node, ":", outreach, ",", surrender_value)
    random.seed()
    r = random.random() * weight
    weights = np.array([1, 1, 0.0, r, 1.0])
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
    
    # print("INPUT")
    # print("-----")
    # print("Adjacency matrix: ", adjacency_matrix)
    # print("Kingdom names: ", list_of_kingdom_names)

    N = len(adjacency_matrix)   # number of vertices
    closed_walk = []
    conquered_kingdoms = set()

    shortest_distance = [[float("inf") for i in range(N)] for j in range(N)]
    heuristic_value = [[0 for i in range(N)] for j in range(N)]

    G = nx.Graph()
    edges = []

    for i, row in enumerate(adjacency_matrix):
        for j, item in enumerate(adjacency_matrix[i]):
            weight = adjacency_matrix[i][j]
            if weight == 'x':
                continue

            edges.append((i, j, weight))
    
    G.add_weighted_edges_from(edges)

    #shortest_paths = nx.floyd_warshall(G)
    shortest_paths, pred = floyd_warshall(G)

    # pdb.set_trace()

    # for a in range(len(shortest_paths)):
    #     for b in range(len(shortest_paths)):
    #         sys.stdout.write(str(shortest_paths[a][b]) + " ")

    #     print()

    # print()

    # for a in range(len(sp)):
    #     for b in range(len(sp)):
    #         sys.stdout.write(str(sp[a][b]) + " ")

    #     print()

    vertex_sets = [None] * len(G.nodes())
    costs = []
    start_vertex = list_of_kingdom_names.index(starting_kingdom)

    #pdb.set_trace()

    for i, vertex in enumerate(sorted([a for a in G.nodes()])):
        vertex_sets[i] = []
        costs.append(acquire_cost(G, start_vertex, vertex, shortest_paths))
        for adjacent_vertex in G[vertex]:
            vertex_sets[i].append(adjacent_vertex)

    # pdb.set_trace()

    N = 0
    graph_weight = calc_weight(G)
    average_cost = 0.0

    selected, cost, tour = weightedsetcover(G, vertex_sets, costs, start_vertex, shortest_paths, pred, graph_weight, heuristic_value, False)
    #print("Real cost:", total_cost(G, tour, selected, heuristic_value, None))

    average_cost = total_cost(G, tour, selected, heuristic_value, None)
    best_cost = average_cost
    best_tour = tour

    print("Normal:", best_cost)

    start = time.time()
    end = start

    while N < 5000 and ((end - start) < 60 * 3):
        global alpha
        alpha = 0.3
        selected, cost, tour = weightedsetcover(G, vertex_sets, costs, start_vertex, shortest_paths, pred, graph_weight, heuristic_value, True)
        cost = total_cost(G, tour, selected, heuristic_value, average_cost)
        #print("Real cost:", cost)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

        print(N)
        #print("New best:", best_cost, "-", best_tour)
        
        N += 1

        end = time.time()

    print("With N:", N, "new best:", best_cost)

    kingdoms_list = []
    conquer_list = []

    for index in tour:
        kingdoms_list.append(list_of_kingdom_names[index])

    for index in selected:
        conquer_list.append(list_of_kingdom_names[index])

    #print(kingdoms_list)
    #print(conquer_list)

    # simulated_annealing(G, tour, start_vertex, selected)

    return kingdoms_list, conquer_list

def simulated_annealing(G, tour, start_node, selected):
    # for i in range(0, 100):
    #     vertex_1 = int(random.random() * len(tour))
    #     vertex_2 = int(random.random() * len(tour))

    #     while not(vertex_1 != vertex_2 and tour[vertex_1] != start_node and tour[vertex_2] != start_node):
    #         vertex_1 = int(random.random() * len(tour))
    #         vertex_2 = int(random.random() * len(tour))

    #     tour[vertex_1:vertex_2] = reversed(tour[vertex_1:vertex_2])

    #     vertex_start = tour[min(vertex_1, vertex_2) - 1]
    #     vertex_end = tour[max(vertex_1, vertex_2) + 1]
        
    #     ###########
    #     path_from_start = path(vertex_start, min(vertex_1, vertex_2), pred)

    #     pdb.set_trace()

    #     print("Tour:", tour, "cost:", total_cost(G, tour, set(selected)))

    return None

def weightedsetcover(G, S, costs, start_node, shortest_paths, pred, weight, heuristic_value, randomness):
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
    w = weight
    K = len(S)
    udict = {}
    selected = list()
    adj = [] # During the process, S will be modified. Make a copy for S.

    conquer_path = [start_node]

    for index, item in enumerate(S):
        adj.append(set(item))
        for j in item:
            if j not in udict:
                udict[j] = set()
            udict[j].add(index)

    conquered_kingdoms = set()
    unconquered_kingdoms = set([i for i in range(K)])
    current_node = start_node

    # print("PQ INITIALIZATION")
    # print("-----------------")

    pq = priorityqueue.PriorityQueue()
    cost = 0
    coverednum = 0
    for index, node in enumerate(adj): # add all sets to the priorityqueue
        if len(node) == 0:
            pq.addtask(index, MAXPRIORITY)
            # print("Added node", index, "with priority", MAXPRIORITY)
        else:
            priority = float(costs[index]) / value(G, start_node, index, unconquered_kingdoms, conquered_kingdoms, start_node, w, heuristic_value, randomness)
            pq.addtask(index, priority)
            # print("Added node", index, "with priority", priority, "(cost:", float(costs[index]), ", value:", value(G, start_node, index, unconquered_kingdoms, conquered_kingdoms, start_node), ")")

    # print(adj)

    while len(conquered_kingdoms) < K:
        target_node = pq.poptask() # get the most cost-effective set
        # print("Conquering node: ", target_node)
        path_to_add = path(current_node, target_node, pred)[1:]
        conquer_path.extend(path_to_add)

        current_node = target_node
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
            pq.addtask(adjacent_vertex, MAXPRIORITY)
            #print("Adding task", adjacent_vertex, "with priority", MAXPRIORITY)
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
            pq.addtask(vertex, cost / value(G, target_node, vertex, unconquered_kingdoms, conquered_kingdoms, start_node, w, heuristic_value, randomness))


        adj[target_node].clear()
        pq.addtask(target_node, MAXPRIORITY)

    path_to_add = path(current_node, start_node, pred)[1:]
    conquer_path.extend(path_to_add)
                        
    return selected, cost, conquer_path


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
