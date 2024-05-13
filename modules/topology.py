import numpy as np


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