import numpy as np
import networkx as nx


def response_time_func(C, tau, L_f, L_o, E_X, T_0, f, N, h):
    """
    The `response_time_func` function calculates the maximum response time of a distributed system.
    It uses various parameters related to channel capacity, latency, packet length,
    execution time, and other factors to determine the total time required for processing 
    and communication.

    Args:
        C: Channel capacity.
        tau: Channel latency.
        L_f: Length of the incoming packet.
        L_o: Length of the outgoing packet.
        E_X: Mean task execution time.
        T_0: Initial task overhead.
        f: Overhead factor.
        N: Number of tasks or processes.
        h: Array of latency coefficients.

    Returns:
        response_time: The calculated maximum response time.
        theta: The sum of the execution times of the tasks.
    """

    # Take the first N elements of h.
    h = h[:N]
    # Calculate the mean execution time per task.
    E_X_i = E_X / N
    # Divide the length of the incoming packet by the number of tasks.
    input_data = L_f / N
    # Calculate the overhead of the incoming packet.
    overhead_input = input_data * f
    # Minimum value for the uniform generation of the outgoing packet.
    low = 0.0
    # Maximum value for the uniform generation of the outgoing packet.
    high = (2 * L_o) / N
    # Calculate the round-trip time for each task.
    T_i = 2 * tau * h
    # Calculate the inverse of the time T_i.
    one_over_T_i = 1 / T_i
    # Sum of the inverses of the times T_i.
    sum_one_over_T_j = np.sum(one_over_T_i)
    # Generate exponential execution times.
    X_i = np.random.exponential(scale=E_X_i, size=N)
    # Calculate the total execution time for each task.
    task_time = T_0 + X_i
    # Sum of the execution times of all tasks.
    theta = np.sum(task_time)
    # Generate lengths of the outgoing packets.
    L_o_i = np.random.uniform(low=low, high=high, size=N)
    # Calculate the overhead of the outgoing packets.
    overhead_output = L_o_i * f
    # Calculate the throughput.
    tp = C * (one_over_T_i / sum_one_over_T_j)
    # Calculate the forward time.
    forward_time = (input_data + overhead_input) / tp   
    # Calculate the return time.
    return_time = np.divide(np.add(L_o_i, overhead_output), tp)   
    # Sum of the forward time and the execution time.
    r_t_1 = np.add(forward_time, task_time)  
    # Sum of the return time to the previous total.
    r_t = np.add(r_t_1, return_time)
    # Find the maximum response time.
    response_time = np.max(r_t)

    # Return the maximum response time and the sum of the execution times.
    return (response_time, theta)



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
