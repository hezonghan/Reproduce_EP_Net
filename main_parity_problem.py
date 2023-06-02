

import json
import math

from gmp import DynamicGMP, MutationModifiedBackPropagation, MutationModifiedBackPropagation_Focus, MutationSimulatedAnnealing, MutationNodeSplitting
from ep_net import ep_net_optimize, simple_gmp_fitness, simple_rank_based_selection, DefaultInitializeMutationController, DefaultEvolveMutationController
from hzh_utils import get_date_time_str


def generate_bitstring(v, input_values=[0.0, 1.0]):
    return [
        # 1 if v & (2**b) > 0 else 0
        # 1 if v & (2**b) > 0 else -1
        input_values[1] if v & (2**b) > 0 else input_values[0]
        for b in range(0, N)  # result[0] is the lowest bit
    ]


def generate_dataset(N, with_bias=False, input_values=[0.0, 1.0], output_values=[0.0, 1.0]):
    return [
        {
            'input': generate_bitstring(i, input_values) if (not with_bias) else generate_bitstring(i, input_values) + [1],
            # 'output': [sum(generate_bitstring(i)) % 2],
            # 'output': [0.4 + 0.2 * (sum(generate_bitstring(i)) % 2)],
            'output': [output_values[round(sum(generate_bitstring(i))) % 2]],
        }
        for i in range(0, 2**N)
    ]


def calculate_hits(gmp:DynamicGMP):
    hits = 0
    for datapoint in dataset:
        estimated_output_data, loss, fpd, _, _ = gmp.evaluate(datapoint['input'], datapoint['output'], derivative_unnecessary=True)
        if (estimated_output_data[0] - 0.5) * (datapoint['output'][0] - 0.5) > 0: hits += 1
    return hits


if __name__ == '__main__':

    start_date_time_str = get_date_time_str()
    print('\n\nStarted at {}'.format(start_date_time_str))

    with_bias = True
    N = 4
    input_values = [0.0, +1.0]
    output_values = [0.2, 0.8]
    dataset = generate_dataset(N, with_bias, input_values, output_values)
    # print(json.dumps(dataset, indent=4))
    print('\nSolving N-parity problem : \033[1;32mN={}\033[0m'.format(N))

    # print('Approximate time of population initialization:\n\tN=3\t8s\n\tN=4\t20s\n\tN=6')
    print('Approximate time of population initialization:\n\tN=3\t14s\n\tN=4\t54s\n\tN=5\t180s')

    # ==================================================
    
    # parameters specified in the original paper:

    parent_selection_function = simple_rank_based_selection

    population_size=20
    lb_init_hidden_nodes_cnt, ub_init_hidden_nodes_cnt = 2,N
    init_conn_density = 0.75
    mbp_init_learning_rate = 0.5
    mbp_lb_learning_rate, mbp_ub_learning_rate = 0.1, 0.6

    mbp_once_total_epochs = 100
    lb_deleted_nodes_cnt, ub_deleted_nodes_cnt, lb_added_nodes_cnt, ub_added_nodes_cnt = 1, 2, 1, 2  # not sure
    lb_deleted_connections_cnt, ub_deleted_connections_cnt, lb_added_connections_cnt, ub_added_connections_cnt = 1, 3, 1, 3  # not sure
    sa_number_of_temperature = 5
    sa_iterations_per_temperature = 100

    # ------------------------------

    # parameters specified in the original paper, but modified by HZH:

    fitness_function = simple_gmp_fitness

    # ------------------------------

    # parameters not specified, and tried by HZH:


    # lb_hidden, ub_hidden = 1, 10
    # lb_hidden, ub_hidden = 10, 20
    # lb_hidden, ub_hidden = 5, 10
    # lb_hidden, ub_hidden = 2, 10
    lb_hidden, ub_hidden = 2, N
    # lb_hidden, ub_hidden = 2, (N//2)+1
    # -------
    # lb_init_hidden_nodes_cnt, ub_init_hidden_nodes_cnt = lb_hidden, ub_hidden  # overwritten
    # lb_init_hidden_nodes_cnt, ub_init_hidden_nodes_cnt = 2, (N//2)+1  # overwritten
    # -------
    # init_weight_abs_ub = 1
    init_weight_abs_ub = 10
    # init_weight_abs_ub = 100
    # init_weight_abs_ub = 30


    # mbp_init_learning_rate = 0.5  # overwritten
    # mbp_init_learning_rate = 0.005  # overwritten
    mbp_init_learning_rate = 1.5  # overwritten
    # -------
    # mbp_lb_learning_rate, mbp_ub_learning_rate = 0.1, 0.6  # overwritten
    # mbp_lb_learning_rate, mbp_ub_learning_rate = 0.001, 0.006  # overwritten
    mbp_lb_learning_rate, mbp_ub_learning_rate = 0.001, 10  # overwritten
    # mbp_lb_learning_rate, mbp_ub_learning_rate = 0.01, 10  # overwritten
    # -------
    # mbp_learning_rate_change = 0.05
    # mbp_learning_rate_change = 0.0005
    # learning_rate_increase_multiple, learning_rate_decrease_multiple = 1.05, 0.6
    learning_rate_increase_multiple, learning_rate_decrease_multiple = 1.25, 0.8
    # learning_rate_increase_multiple, learning_rate_decrease_multiple = 2, 0.5
    # -------
    # mbp_once_total_epochs = 300  # overwritten
    mbp_once_total_epochs = 3000  # overwritten
    # -------
    mbp_learning_rate_adapt_epochs = 5


    sa_temperatures_list = [
        -0.10 / math.log(0.5),
        -0.08 / math.log(0.5),
        -0.06 / math.log(0.5),
        -0.04 / math.log(0.5),
        -0.02 / math.log(0.5),
    ]  #  - loss_increase / temp = ln P_accept = ln 0.5
    # -------
    # sa_iterations_per_temperature = 0  # overwritten


    node_spliting_alpha = 0.4
    # node_spliting_alpha = -0.2


    # ub_deleted_nodes_cnt = 1
    # ub_added_nodes_cnt = 1
    # ub_deleted_connections_cnt = 1
    # ub_added_connections_cnt = 1
    # overwritten


    # generations = 100
    generations = 1000
    evolve_significant_reduce_ratio = 0.05  # reduced to (under) 95% of the parent will be considered as "significantly reduced".

    # ==================================================

    mbp = MutationModifiedBackPropagation(
    # mbp = MutationModifiedBackPropagation_Focus(
        init_learning_rate=mbp_init_learning_rate,
        # learning_rate_change=mbp_learning_rate_change,
        learning_rate_increase_multiple=learning_rate_increase_multiple,
        learning_rate_decrease_multiple=learning_rate_decrease_multiple,
        lb_learning_rate=mbp_lb_learning_rate,
        ub_learning_rate=mbp_ub_learning_rate,
        learning_rate_adapt_epochs=mbp_learning_rate_adapt_epochs,
        total_epochs=mbp_once_total_epochs,
    )
    sa = MutationSimulatedAnnealing(
        fitness_function=fitness_function,
        temperatures_list=sa_temperatures_list,
        iterations_per_temperature=sa_iterations_per_temperature,
    )
    split = MutationNodeSplitting(
        alpha=node_spliting_alpha,
    )

    # ------------------------------

    initialize_mutation_controller = DefaultInitializeMutationController(mbp=mbp)

    evolve_mutation_controller = DefaultEvolveMutationController(
        fitness_function=fitness_function,
        significant_reduce_ratio=evolve_significant_reduce_ratio,
        mbp=mbp,
        sa=sa,
        split=split,
        lb_hidden=lb_hidden,
        ub_hidden=ub_hidden,
        lb_deleted_nodes_cnt=lb_deleted_nodes_cnt,
        ub_deleted_nodes_cnt=ub_deleted_nodes_cnt,
        lb_deleted_connections_cnt=lb_deleted_connections_cnt,
        ub_deleted_connections_cnt=ub_deleted_connections_cnt,
        lb_added_nodes_cnt=lb_added_nodes_cnt,
        ub_added_nodes_cnt=ub_added_nodes_cnt,
        lb_added_connections_cnt=lb_added_connections_cnt,
        ub_added_connections_cnt=ub_added_connections_cnt,
    )

    # ------------------------------

    best_gmp:DynamicGMP
    best_gmp, used_generations = ep_net_optimize(
        dataset=dataset,

        lb_hidden=lb_hidden,
        ub_hidden=ub_hidden,
        lb_init_hidden_nodes_cnt=lb_init_hidden_nodes_cnt,
        ub_init_hidden_nodes_cnt=ub_init_hidden_nodes_cnt,
        init_conn_density=init_conn_density,
        init_weight_abs_ub=init_weight_abs_ub,
        
        population_size=population_size,
        generations=generations,
        fitness_function=fitness_function,
        parent_selection_function=parent_selection_function,

        initialize_mutation_controller=initialize_mutation_controller,
        evolve_mutation_controller = evolve_mutation_controller,
    )

    print('mbp epochs: {}'.format(mbp.all_invokes_epochs_cnt))

    print('\n\033[1;32mBest gmp:\033[0m\n')
    print('nodes: {}'.format( ' -> '.join([str(node_id) for node_id in best_gmp.next_nodes(0)]) ))
    # print(json.dumps(best_gmp.conn, indent=4))
    best_gmp.display()

    best_gmp_hits = calculate_hits(best_gmp)

    print('\n\nSolved N-parity problem : \033[1;32mN={}\033[0m\n'.format(N))

    result_file_path = './results/{}__N={}__{:04d}gen__hit{:04d}.json'.format(start_date_time_str, N, used_generations, best_gmp_hits)
    print('\n\nResult saved at \033[34m{}\033[0m'.format(result_file_path))
    result_json = {
        'result_file_path': result_file_path,
        'start_date_time_str': start_date_time_str,
        'problem': {
            'N': N,
            'with_bias': with_bias,
            'input_values': input_values,
            'output_values': output_values,
        },
        'config': {
            'evolve': {
                'population_size': population_size,
                'generations': generations,
                'evolve_significant_reduce_ratio': evolve_significant_reduce_ratio,
            },
            'network_structure_limit': {
                'lb_hidden': lb_hidden,
                'ub_hidden': ub_hidden,

                'lb_deleted_nodes_cnt': lb_deleted_nodes_cnt,
                'ub_deleted_nodes_cnt': ub_deleted_nodes_cnt,
                'lb_added_nodes_cnt': lb_added_nodes_cnt,
                'ub_added_nodes_cnt': ub_added_nodes_cnt,
                'lb_deleted_connections_cnt': lb_deleted_connections_cnt,
                'ub_deleted_connections_cnt': ub_deleted_connections_cnt,
                'lb_added_connections_cnt': lb_added_connections_cnt,
                'ub_added_connections_cnt': ub_added_connections_cnt,
            },
            'network_structure_init': {
                'lb_init_hidden_nodes_cnt': lb_init_hidden_nodes_cnt,
                'ub_init_hidden_nodes_cnt': ub_init_hidden_nodes_cnt,
                'init_conn_density': init_conn_density,
                'init_weight_abs_ub': init_weight_abs_ub,
            },
            'mbp': {
                'mbp_init_learning_rate': mbp_init_learning_rate,
                'mbp_lb_learning_rate': mbp_lb_learning_rate,
                'mbp_ub_learning_rate': mbp_ub_learning_rate,
                'learning_rate_increase_multiple': learning_rate_increase_multiple,
                'learning_rate_decrease_multiple': learning_rate_decrease_multiple,
                'mbp_once_total_epochs': mbp_once_total_epochs,
                'mbp_learning_rate_adapt_epochs': mbp_learning_rate_adapt_epochs,
            },
            'sa': {
                'sa_number_of_temperature': sa_number_of_temperature,
                'sa_iterations_per_temperature': sa_iterations_per_temperature,
                'sa_temperatures_list': sa_temperatures_list,
            },
            'node_spliting': {
                'node_spliting_alpha': node_spliting_alpha,
            },
        },
        'result': {
            'used_generations': used_generations,
            'hits': best_gmp_hits,
            'network': {
                'd_input': best_gmp.d_input,
                'd_output': best_gmp.d_output,
                'lb_hidden': best_gmp.lb_hidden,
                'ub_hidden': best_gmp.ub_hidden,

                'nodes_cnt': best_gmp.nodes_cnt,
                'active_hidden_nodes_cnt': best_gmp.active_hidden_nodes_cnt,

                'prev_node': best_gmp.prev_node,
                'next_node': best_gmp.next_node,
                'conn': best_gmp.conn,
            },
        },
    }
    f = open(result_file_path, 'x')
    f.write(json.dumps(result_json, indent=4))
    f.close()
