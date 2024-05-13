import numpy as np
import networkx as nx


def response_time_func(C, tau, L_f, L_o, E_X, T_0, f, N, h):
    
    h = h[:N]
    
    E_X_i = E_X/N 
    input_data = L_f/N
    overhead_input = input_data*f
    
    low = 0.0                 
    high = (2*L_o)/N
    
    #T_i
    T_i = 2 * tau * h
    one_over_T_i = 1/T_i
    sum_one_over_T_j = np.sum(one_over_T_i) 
    
    #tempo esecuzione
    X_i = np.random.exponential(scale = E_X_i, size = N)
    task_time = T_0 + X_i
    theta = np.sum(task_time)
    
    #Valutazione size output
    L_o_i = np.random.uniform(low = low, high = high, size = N)
    
    #Overhead output
    overhead_output = L_o_i * f 
    
    #throughput
    tp = C * (one_over_T_i/sum_one_over_T_j)
    
    #valutazione tempo andata
    forward_time = (input_data+overhead_input)/tp
    
    #valutazione tempo ritorno
    return_time = np.divide(np.add(L_o_i,overhead_output),tp)
    
    r_t_1 = np.add(forward_time, task_time)
    r_t = np.add(r_t_1, return_time)
    
    response_time = np.max(r_t)
    
    return(response_time, theta)


def construct_jellyfish_topology(switches, ports, r, n_servers) -> nx.Graph:
    """
    Build a Jellyfish topology graph.

    Args:
        switches: The number of switches in the Jellyfish topology.
        ports: The number of ports per switch.
        r: The number of neighbor switches to connect to.
        n_servers: The total number of servers to add to the topology.

    Returns:
        A Jellyfish topology graph with the specified parameters.
    """

    #Create switch structure
    G = nx.random_regular_graph(r, switches)

    #Label switches with 0
    for i in range(switches):
        G.nodes[i]['type'] = 0 
    #Add servers labeling them as 1
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(G.edges())
    servers_counter = 0
    for i in range(switches):
        for j in range(ports - r):
            if servers_counter <= n_servers:
                server_name = f'server_{i}_{j}'
                H.add_node(server_name)
                H.nodes[server_name]['type'] = 1
                H.add_edge(i, server_name)
                servers_counter += 1 
    return H
