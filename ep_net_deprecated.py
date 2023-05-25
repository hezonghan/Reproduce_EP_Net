


# def default_initial_mutation_function(
#         gmp:DynamicGMP, dataset, 
#         mbp:MutationModifiedBackPropagation
#     ):
#     mbp.operate(gmp, dataset)


# def default_evolve_mutation_function(
#         gmp:DynamicGMP, dataset,
#         fitness_function, parent_fitness, worst_fitness,
#         mbp:MutationModifiedBackPropagation,
#         sa:MutationSimulatedAnnealing,
#         lb_deleted_nodes_cnt,
#         ub_deleted_nodes_cnt,
#         lb_deleted_connections_cnt,
#         ub_deleted_connections_cnt,
#     ):

#     # fitness_0 = fitness_function(gmp)

#     mbp.operate(gmp, dataset)
#     # --------
#     sa.operate(gmp, dataset)
#     # --------
#     # fitness_1 = fitness_function(gmp)
#     # if fitness_1 < parent_fitness:
#     #     return gmp, 'parent'
#     if fitness_function(gmp) < parent_fitness:
#         return gmp, 'parent'
    

#     deleted_nodes_cnt = random.randint(lb_deleted_nodes_cnt, ub_deleted_nodes_cnt)
#     deleted_nodes = random.sample(list(gmp.hidden_nodes()), deleted_nodes_cnt)
#     for deleted_node_id in deleted_nodes:
#         gmp.delete_hidden_node(deleted_node_id)
#     # --------
#     mbp.operate(gmp, dataset)
#     # --------
#     if fitness_function(gmp) < worst_fitness:
#         return gmp, 'worst'
    

#     deleted_connections_cnt = random.randint(lb_deleted_connections_cnt, ub_deleted_connections_cnt)
#     deleted_connections = random.sample(list(gmp.enabled_edges()), deleted_connections_cnt)
#     for (src_node_id, dst_node_id) in deleted_connections:
#         gmp.delete_connection(src_node_id, dst_node_id)
#     # --------
#     mbp.operate(gmp, dataset)
#     # --------
#     if fitness_function(gmp) < worst_fitness:
#         return gmp, 'worst'
    



class MutationController:

    def __init__(self):
        pass

    def operate(self, gmp:DynamicGMP, dataset):
        raise NotImplementedError



def ep_net_optimize(
        # ---------------------
        # mutators,
        # fitness_function,
        # parent_selection_function,
        # population_size,
        # ---------------------
        population_size,
        fitness_function,
        parent_selection_function,
        # initial_mutators,
        # evolve_mutators,
        initial_mutation_function=default_initial_mutation_function,
        evolve_mutation_function=default_evolve_mutation_function,
):
    for gmp in population: 
        initial_mutation_function(
            gmp=gmp,
            dataset=dataset,

            mbp=,
        )
    


    

    while True:
        selected_parent_idx, worst_idx = parent_selection_function(fitness_values)
        offspring, replaced = evolve_mutation_function(
            gmp=copy.deepcopy(population[selected_parent_idx]),
            dataset=dataset,

            fitness_function=fitness_function,
            parent_fitness=fitness_values[selected_parent_idx],
            worst_fitness=fitness_values[worst_idx],

            mbp=,
            sa=,
            
            lb_deleted_nodes_cnt=,
            ub_deleted_nodes_cnt=,
            lb_deleted_connections_cnt=,
            ub_deleted_connections_cnt=,
        )