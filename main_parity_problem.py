

import json
import math

from gmp import MutationModifiedBackPropagation, MutationModifiedBackPropagation_Focus, MutationSimulatedAnnealing, MutationNodeSplitting
from ep_net import ep_net_optimize, simple_gmp_fitness, simple_rank_based_selection, DefaultInitializeMutationController, DefaultEvolveMutationController
from hzh_utils import get_date_time_str


def generate_bitstring(v):
    return [
        1 if v & (2**b) > 0 else 0
        for b in range(0, N)  # result[0] is the lowest bit
    ]


def generate_dataset(N):
    return [
        {
            'input': generate_bitstring(i),
            'output': [sum(generate_bitstring(i)) % 2],
            # 'output': [0.4 + 0.2 * (sum(generate_bitstring(i)) % 2)],
        }
        for i in range(0, 2**N)
    ]


if __name__ == '__main__':

    print('\n\nStarted at {}'.format(get_date_time_str()))

    N = 3
    dataset = generate_dataset(N)
    # print(json.dumps(dataset, indent=4))
    print('\nSolving N-parity problem : \033[1;32mN={}\033[0m'.format(N))

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
    sa_iterations_per_temperature = 0  # overwritten


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

    print('\n\nSolved N-parity problem : \033[1;32mN={}\033[0m\n'.format(N))
