

import copy
import json
import math
import random

from timeit import default_timer

from gmp import DynamicGMP, MutationModifiedBackPropagation, MutationSimulatedAnnealing, MutationNodeSplitting


def simple_gmp_fitness(gmp:DynamicGMP, dataset):  # not the same as described in the paper: loss is not squared.
    overall_loss, _ = gmp.evaluate_all(dataset, derivative_unnecessary=True)
    return overall_loss / len(dataset) / gmp.d_output


def simple_rank_based_selection(fitness_values):
    population_size = len(fitness_values)
    rank_to_idx = list(range(0, population_size))
    rank_to_idx.sort(key=lambda idx: fitness_values[idx])

# def simple_rank_based_selection(sorted_fitness_values):
#     population_size = len(sorted_fitness_values)

    # Weights for simple rank-based selection (where P refers to population size):
    # rank == 0   (best ): weight = P
    # rank == P-1 (worst): weight = 1
    # rank >= rank_mid   : weight = (1 + 2 + ... + (P - rank_mid)) = (P - rank_mid + 1) * (P - rank_mid) / 2.0
    weight_sum = (population_size + 1) * population_size // 2

    target_weight = random.random() * weight_sum
    rank_lo, rank_hi = 0, (population_size - 1)

    while rank_lo <= rank_hi:  # binary search
        rank_mid = (rank_lo + rank_hi) // 2
        weight_mid_lb = (population_size - rank_mid) * (population_size - rank_mid - 1) // 2  # rank >  rank_mid
        weight_mid_ub = (population_size - rank_mid) * (population_size - rank_mid + 1) // 2  # rank >= rank_mid
        if weight_mid_lb <= target_weight <= weight_mid_ub:
            return rank_to_idx[rank_mid], rank_to_idx[-1]
            # return rank_mid
        elif target_weight < weight_mid_lb:
            rank_lo = rank_mid + 1
        else:
            rank_hi = rank_mid - 1
    assert False


def test_connection_importance(gmp:DynamicGMP, dataset):
    xi = {
        node_i_id: 
        {
            node_j_id:
            {
                'arr': [],
                # 'sum': 0,
                'importance': 0,
            }
            # for node_j_id in gmp.next_nodes(node_i_id, including=False)
            # if gmp.is_enabled_edge(node_i_id, node_j_id)  # also calculated importance for disabled connection
            # for node_j_id in gmp.conn[node_i_id].keys()
            for node_j_id in gmp.next_nodes(node_i_id, including=False)  if gmp.is_existing_edge(node_i_id, node_j_id)
        }
        # for node_i_id in gmp.next_nodes(0)
        # for node_i_id in gmp.conn.keys()
        for node_i_id in gmp.next_nodes(0)
    }
    for datapoint in dataset:
        _, _, fpd, _, _ = gmp.evaluate(datapoint['input'], datapoint['output'])

        for node_i_id in xi.keys():
            for node_j_id in xi[node_i_id].keys():
                
                # for debug:
                if (node_i_id not in fpd) or (node_j_id not in fpd[node_i_id]):
                    print('\t\t')
                    # print('\t\tnodes: {}'.format( ' -> '.join(list(gmp.next_nodes(0))) ))
                    print('\t\tnodes: {}'.format( ' -> '.join([str(node_id) for node_id in gmp.next_nodes(0)]) ))
                    print('\t\tCalculating xi[node_i_id={}][node_j_id={}]'.format(node_i_id, node_j_id))
                    print('\t\tfpd:\n{}'.format(json.dumps(fpd, indent=4)))
                    print('\t\t')

                current_weight = gmp.conn[node_i_id][node_j_id][1] if gmp.conn[node_i_id][node_j_id][0] else 0
                updated_weight = current_weight + fpd[node_i_id][node_j_id]
                xi[node_i_id][node_j_id]['arr'].append(updated_weight)

    for node_i_id in xi.keys():
        for node_j_id in xi[node_i_id].keys():
            xi_arr = xi[node_i_id][node_j_id]['arr']
            xi_sum = sum(xi_arr)
            xi_avg = xi_sum / len(dataset)
            xi_imp_denominator = math.sqrt(sum([ (xi_value - xi_avg) ** 2  for xi_value in xi_arr ]))
            if xi_imp_denominator < 1e-5: xi_imp_denominator = 1e-5  # avoid divide by zero
            xi_imp = xi_sum / xi_imp_denominator
            xi[node_i_id][node_j_id]['importance'] = xi_imp
    
    return xi


class InitializeMutationController:

    def __init__(self):
        pass

    def operate(self, gmp:DynamicGMP, dataset):
        raise NotImplementedError


class DefaultInitializeMutationController(InitializeMutationController):

    def __init__(
            self, 
            mbp:MutationModifiedBackPropagation
        ):
        super().__init__()
        self.mbp = mbp

    def operate(self, gmp:DynamicGMP, dataset):
        self.mbp.operate(gmp, dataset)


class EvolveMutationController:

    def __init__(self):
        pass

    def operate(self, gmp:DynamicGMP, dataset, parent_fitness, worst_fitness):
        raise NotImplementedError


class DefaultEvolveMutationController(EvolveMutationController):

    def __init__(
            self,
            fitness_function,
            significant_reduce_ratio,
            mbp:MutationModifiedBackPropagation,
            sa:MutationSimulatedAnnealing,
            split:MutationNodeSplitting,
            lb_hidden,
            ub_hidden,
            lb_deleted_nodes_cnt,
            ub_deleted_nodes_cnt,
            lb_deleted_connections_cnt,
            ub_deleted_connections_cnt,
            lb_added_nodes_cnt,
            ub_added_nodes_cnt,
            lb_added_connections_cnt,
            ub_added_connections_cnt,
        ):
        super().__init__()

        self.fitness_function = fitness_function
        self.significant_reduce_ratio = significant_reduce_ratio

        self.mbp = mbp
        self.sa = sa
        self.split = split

        self.lb_hidden = lb_hidden
        self.ub_hidden = ub_hidden

        self.lb_deleted_nodes_cnt = lb_deleted_nodes_cnt
        self.ub_deleted_nodes_cnt = ub_deleted_nodes_cnt
        self.lb_deleted_connections_cnt = lb_deleted_connections_cnt
        self.ub_deleted_connections_cnt = ub_deleted_connections_cnt
        self.lb_added_nodes_cnt = lb_added_nodes_cnt
        self.ub_added_nodes_cnt = ub_added_nodes_cnt
        self.lb_added_connections_cnt = lb_added_connections_cnt
        self.ub_added_connections_cnt = ub_added_connections_cnt

    def operate(self, gmp:DynamicGMP, dataset, parent_fitness, worst_fitness):

        self.mbp.operate(gmp, dataset)
        # --------
        self.sa.operate(gmp, dataset)
        # --------
        if self.fitness_function(gmp, dataset) < parent_fitness * (1 - self.significant_reduce_ratio):
            print('\treturned by A')
            return gmp, 'parent'


        conn_importance = test_connection_importance(gmp, dataset)
        enabled_conn = sorted(
            [
                (node_i_id, node_j_id)
                for node_i_id in conn_importance
                for node_j_id in conn_importance[node_i_id]
                if gmp.is_enabled_edge(node_i_id, node_j_id)
            ],
            # key=(lambda node_i_id, node_j_id: conn_importance[node_i_id][node_j_id]['importance']),
            key=(lambda tp: conn_importance[tp[0]][tp[1]]['importance']),
            reverse=True
        )
        disabled_conn = sorted(
            [
                (node_i_id, node_j_id)
                for node_i_id in conn_importance
                for node_j_id in conn_importance[node_i_id]
                if not gmp.is_enabled_edge(node_i_id, node_j_id)
            ],
            # key=(lambda node_i_id, node_j_id: conn_importance[node_i_id][node_j_id]['importance']),
            key=(lambda tp: conn_importance[tp[0]][tp[1]]['importance']),
            reverse=True
        )
        

        # deleted_nodes_cnt = random.randint(self.lb_deleted_nodes_cnt, self.ub_deleted_nodes_cnt)
        # # if deleted_nodes_cnt <= gmp.active_hidden_nodes_cnt:
        # # if gmp.active_hidden_nodes_cnt - deleted_nodes_cnt >= 1:
        # if gmp.active_hidden_nodes_cnt - deleted_nodes_cnt >= self.lb_hidden:
        
        temporary_ub_deleted_nodes_cnt = min(gmp.active_hidden_nodes_cnt - self.lb_hidden, self.ub_deleted_nodes_cnt)
        if self.lb_deleted_nodes_cnt <= temporary_ub_deleted_nodes_cnt:
            deleted_nodes_cnt = random.randint(self.lb_deleted_nodes_cnt, temporary_ub_deleted_nodes_cnt)

            deleted_nodes_gmp = copy.deepcopy(gmp)
            deleted_nodes = random.sample(list(deleted_nodes_gmp.all_hidden_nodes()), deleted_nodes_cnt)
            for deleted_node_id in deleted_nodes:
                deleted_nodes_gmp.delete_hidden_node(deleted_node_id)
            # --------
            self.mbp.operate(deleted_nodes_gmp, dataset)
            # --------
            if self.fitness_function(deleted_nodes_gmp, dataset) < worst_fitness:
                print('\treturned by B')
                return deleted_nodes_gmp, 'worst'

        
        # deleted_connections_cnt = random.randint(self.lb_deleted_connections_cnt, self.ub_deleted_connections_cnt)
        # if deleted_connections_cnt <= len(enabled_conn):

        temporary_ub_deleted_connections_cnt = min(len(enabled_conn), self.ub_deleted_connections_cnt)
        if self.lb_deleted_connections_cnt <= temporary_ub_deleted_connections_cnt:
            deleted_connections_cnt = random.randint(self.lb_deleted_connections_cnt, temporary_ub_deleted_connections_cnt)

            deleted_connections_gmp = copy.deepcopy(gmp)
            # deleted_connections = random.sample(list(deleted_connections_gmp.enabled_edges()), deleted_connections_cnt)  # TODO  sample according to test function
            deleted_connections = enabled_conn[-deleted_connections_cnt:]  # delete the most UN-important connections.
            for (src_node_id, dst_node_id) in deleted_connections:
                deleted_connections_gmp.disable_connection(src_node_id, dst_node_id)
            # --------
            self.mbp.operate(deleted_connections_gmp, dataset)
            # --------
            if self.fitness_function(deleted_connections_gmp, dataset) < worst_fitness:
                print('\treturned by C')
                return deleted_connections_gmp, 'worst'
        
        
        # added_nodes_cnt = random.randint(self.lb_added_nodes_cnt, self.ub_added_nodes_cnt)
        # if (added_nodes_cnt <= gmp.active_hidden_nodes_cnt) and (gmp.active_hidden_nodes_cnt + added_nodes_cnt <= self.ub_hidden):
        
        temporary_ub_added_nodes_cnt = min([self.ub_hidden - gmp.active_hidden_nodes_cnt, gmp.active_hidden_nodes_cnt, self.ub_added_nodes_cnt])
        if self.lb_added_nodes_cnt <= temporary_ub_added_nodes_cnt:
            added_nodes_cnt = random.randint(self.lb_added_nodes_cnt, temporary_ub_added_nodes_cnt)

            added_nodes_gmp = copy.deepcopy(gmp)
            splitted_nodes = random.sample(list(added_nodes_gmp.all_hidden_nodes()), added_nodes_cnt)
            for splitted_node_id in splitted_nodes:
                self.split.operate(added_nodes_gmp, original_node_id=splitted_node_id)
            # --------
            self.mbp.operate(added_nodes_gmp, dataset)
            # --------
            if self.fitness_function(added_nodes_gmp, dataset) < worst_fitness:
                print('\treturned by D')
                return added_nodes_gmp, 'worst'
        
        
        # added_connections_cnt = random.randint(self.lb_added_connections_cnt, self.ub_added_connections_cnt)
        # if added_connections_cnt <= len(disabled_conn):
        
        temporary_ub_added_connections_cnt = min(len(disabled_conn), self.ub_added_connections_cnt)
        if self.lb_added_connections_cnt <= temporary_ub_added_connections_cnt:
            added_connections_cnt = random.randint(self.lb_added_connections_cnt, temporary_ub_added_connections_cnt)

            added_connections_gmp = copy.deepcopy(gmp)
            added_connections = disabled_conn[:added_connections_cnt]  # add the most important connections.
            for (src_node_id, dst_node_id) in added_connections:
                added_connections_gmp.enable_connection(src_node_id, dst_node_id)
            # --------
            self.mbp.operate(added_connections_gmp, dataset)
            # --------
            if self.fitness_function(added_connections_gmp, dataset) < worst_fitness:
                print('\treturned by E')
                return added_connections_gmp, 'worst'
    
        print('\treturned by F')
        return gmp, 'none'





    


class EPNet:
    # steady-state, mutation-only

    def __init__(self, mutators):
        self.mutators = mutators

        self.population = []

    def run():
        pass

def ep_net_optimize(
        dataset,
        # ---------------------
        lb_hidden,
        ub_hidden,
        lb_init_hidden_nodes_cnt,
        ub_init_hidden_nodes_cnt,
        init_conn_density,
        init_weight_abs_ub,
        # ---------------------
        population_size,
        generations,
        fitness_function,
        parent_selection_function,
        initialize_mutation_controller:InitializeMutationController,
        evolve_mutation_controller    :EvolveMutationController,
    ):
    d_input = len(dataset[0]['input'])
    d_output = len(dataset[0]['output'])

    t0 = default_timer()

    population = [
        DynamicGMP(
            d_input=d_input,
            d_output=d_output,
            lb_hidden=lb_hidden,
            ub_hidden=ub_hidden,
            init_hidden_nodes_cnt=random.randint(lb_init_hidden_nodes_cnt, ub_init_hidden_nodes_cnt),
            init_conn_density=init_conn_density,
            init_weight_abs_ub=init_weight_abs_ub,
        )
        for _ in range(0, population_size)
    ]
    for gmp in population: 
        initialize_mutation_controller.operate(gmp=gmp, dataset=dataset)
    fitness_values = [
        fitness_function(gmp, dataset)
        for gmp in population
    ]
    
    t1 = default_timer()
    print('\nInitial population generating + evaluating costs {:.3f} s.\n'.format(t1 - t0))

    for generation in range(1, generations+1):  # TODO
        print('\ngeneration #{}:'.format(generation))
        # for gmp_idx in range(0, population_size): print('\tfitness={:.6f} , active_hidden={}'.format(fitness_values[gmp_idx], population[gmp_idx].active_hidden_nodes_cnt))

        selected_parent_idx, worst_idx = parent_selection_function(fitness_values)
        offspring, replaced = evolve_mutation_controller.operate(
            gmp=copy.deepcopy(population[selected_parent_idx]),
            dataset=dataset,
            parent_fitness=fitness_values[selected_parent_idx],
            worst_fitness=fitness_values[worst_idx],
        )
        print('\tselected_parent  fitness={:.6f} , active_hidden={}'.format(fitness_values[selected_parent_idx], population[selected_parent_idx].active_hidden_nodes_cnt))
        print('\toffspring        fitness={:.6f} , active_hidden={}'.format(fitness_function(offspring, dataset), offspring.active_hidden_nodes_cnt))
        # print('\treplaced: {}'.format(replaced))
        # print('\treplaced: {} (parent: {} , worst: {})'.format(replaced, fitness_values[selected_parent_idx], fitness_values[worst_idx]))
        print('\treplaced: {} (worst: {:.6f})'.format(replaced, fitness_values[worst_idx]))

        if replaced == 'parent':
            population[selected_parent_idx] = offspring
            fitness_values[selected_parent_idx] = fitness_function(offspring, dataset)
        elif replaced == 'worst':
            population[worst_idx] = offspring
            fitness_values[worst_idx] = fitness_function(offspring, dataset)
        elif replaced == 'none':
            pass
        else:
            assert False

    best_gmp_idx = min(list(range(0, population_size)), key=lambda gmp_idx: fitness_values[gmp_idx])
    best_gmp = population[best_gmp_idx]

    print('best gmp: fitness={:.6f} , active_hidden={}'.format(fitness_values[best_gmp_idx], population[best_gmp_idx].active_hidden_nodes_cnt))

    return best_gmp