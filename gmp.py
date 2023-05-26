

import copy
import json
import math
import random

# from gmp import DynamicGMP

# import torch
# import torch.nn


class DynamicGMP:

    def __init__(
            self, 
            d_input, d_output, lb_hidden, ub_hidden, 
            init_hidden_nodes_cnt=0, init_conn_density=0.75, init_weight_abs_ub=1.0,
            is_deepcopying=False,
        ):
        
        self.d_input = d_input
        self.d_output = d_output
        self.lb_hidden = lb_hidden  # TODO
        self.ub_hidden = ub_hidden  # TODO
        # self.node_splitting_alpha = node_splitting_alpha

        if is_deepcopying: return

        self.nodes_cnt = d_input + d_output + init_hidden_nodes_cnt  # including deleted hidden nodes
        self.active_hidden_nodes_cnt = init_hidden_nodes_cnt

        # _______________________________________

        # Node ID allocation:
        # input :  0                       ...  {d_input} - 1
        # output:  {d_input}               ...  {d_input} + {d_output} - 1
        # hidden:  {d_input} + {d_output}  ...
        # Note that the allocated ID to a hidden node which is later deleted will not be reused.

        # So, the feedforward process traverse in the following order:
        nodes_order = \
            list(range(0, d_input)) + \
            list(range(d_input + d_output, d_input + d_output + init_hidden_nodes_cnt)) + \
            list(range(d_input, d_input + d_output))

        # nodes order, implemented as a linked list:
        self.prev_node = {}
        self.next_node = {}
        for pos in range(0, d_input + d_output + init_hidden_nodes_cnt):
            self.prev_node[nodes_order[pos]] = None if (pos == 0                 ) else nodes_order[pos-1]
            self.next_node[nodes_order[pos]] = None if (pos == self.nodes_cnt - 1) else nodes_order[pos+1]

        # _______________________________________

        # Connections classification:

        # (1) never exists:
        #       is_input_node(dst_node_id)
        #       src_node_id    ==    dst_node_id  // violating the linked list information
        #       src_node_id  behind  dst_node_id  // violating the linked list information
        
        # (2) currently not exists:
        #       is_deleted_hidden_node(src_node_id)
        #       is_deleted_hidden_node(dst_node_id)
        #   (Warning: information may remain in the self.conn. So do NOT use something like "self.conn.keys()" as iterator)

        # (3) exists but disabled:
        #       self.conn[src_node_id][dst_node_id][0] == False
        #   (Note: You may calculate the partial derivative (on some loss functions) of this connection, as if it is enabled BUT having a zero weight)

        # (4) exists and enabled:
        #       self.conn[src_node_id][dst_node_id][0] == True

        self.conn = {
            src_node_id: {
                dst_node_id: [
                    (random.random() <= init_conn_density),  # whether connection enabled
                    init_weight_abs_ub * (2 * random.random() - 1),  # connection weight
                ]
                # for dst_node_id in range(d_input, self.nodes_cnt)
                # for dst_node_id in range(max(d_input, src_node_id + 1), self.nodes_cnt)
                for dst_node_id in self.next_nodes(src_node_id, including=False)  if not self.is_input_node(dst_node_id)
            }
            # for src_node_id in range(0, self.nodes_cnt - 1)  # wrong: because [self.nodes_cnt - 1] is (usually) hidden node rather than output node.
            for src_node_id in self.next_nodes(0)
        }
    
    # =================================================

    def is_input_node(self, node_id):
        return (0 <= node_id < self.d_input)

    def is_output_node(self, node_id):
        return (self.d_input <= node_id < self.d_input + self.d_output)
    
    def is_hidden_node(self, node_id):  # no matter active or deleted
        return (self.d_input + self.d_output <= node_id < self.nodes_cnt)

    def is_deleted_hidden_node(self, node_id):
        if not self.is_hidden_node(node_id): return False
        return (self.prev_node[node_id] is None)
    
    def is_existing_edge(self, src_node_id, dst_node_id):
        if src_node_id not in self.conn: return False
        if dst_node_id not in self.conn[src_node_id]: return False
        # FIXME  to use another "self.conn"-irrelevant approach to check the order.

        if self.is_input_node(dst_node_id): return False

        if self.is_deleted_hidden_node(src_node_id): return False
        if self.is_deleted_hidden_node(dst_node_id): return False

        return True
    
    def is_enabled_edge(self, src_node_id, dst_node_id):
        if not self.is_existing_edge(src_node_id, dst_node_id): return False
        if not self.conn[src_node_id][dst_node_id][0]: return False
        return True
    
    # =================================================

    def previous_nodes(self, node_id, including=True):
        if not including:
            node_id = self.prev_node[node_id]
        while node_id is not None:
            yield node_id
            node_id = self.prev_node[node_id]

    def next_nodes(self, node_id, including=True):
        if not including:
            node_id = self.next_node[node_id]
        while node_id is not None:
            yield node_id
            node_id = self.next_node[node_id]

    # def hidden_nodes(self):
    def all_hidden_nodes(self):
        last_input_node_id = self.d_input - 1
        node_id = self.next_node[last_input_node_id]
        while self.is_hidden_node(node_id):
            yield node_id
            node_id = self.next_node[node_id]

    def all_edges(self, enabled_only):
        for src_node_id in self.next_nodes(0):
            for dst_node_id in self.next_nodes(src_node_id, including=False):
                if not self.is_existing_edge(src_node_id, dst_node_id):
                    continue
                if (not enabled_only) or self.is_enabled_edge(src_node_id, dst_node_id):
                    yield (src_node_id, dst_node_id)

    # =================================================

    def add_hidden_node(self, prev_neighbor_node_id, next_neighbor_node_id):
        assert not self.is_output_node(prev_neighbor_node_id)
        assert not self.is_input_node(next_neighbor_node_id)
        assert self.next_node[prev_neighbor_node_id] == next_neighbor_node_id

        self.nodes_cnt += 1
        new_node_id = self.nodes_cnt - 1
        
        self.active_hidden_nodes_cnt += 1

        # _______________________________________

        # original order: prev_neighbor_node_id           ->          next_neighbor_node_id
        # new order:      prev_neighbor_node_id  ->  new_node_id  ->  next_neighbor_node_id

        self.next_node[prev_neighbor_node_id] = new_node_id
        self.next_node[new_node_id] = next_neighbor_node_id

        self.prev_node[next_neighbor_node_id] = new_node_id
        self.prev_node[new_node_id] = prev_neighbor_node_id

        # _______________________________________
        
        # for debug only:
        # print('\t\tnodes: {}'.format( ' -> '.join([str(node_id) for node_id in self.next_nodes(0)]) ))
        # print('\t\tnew_node_id = {}'.format(new_node_id))
        # print('\t\tconn:\n{}'.format(json.dumps(self.conn, indent=4)))

        self.conn[new_node_id] = {}
        for node_id in self.previous_nodes(prev_neighbor_node_id): self.conn[node_id][new_node_id] = [False, 0.0]# FIXME
        for node_id in self.next_nodes(next_neighbor_node_id):     self.conn[new_node_id][node_id] = [False, 0.0]

        # print('self.conn[new_node_id={}]  =  {}'.format(new_node_id, self.conn[new_node_id]))

        # _______________________________________

        return new_node_id


    # =================================================

    def delete_hidden_node(self, original_node_id):
        assert self.d_input + self.d_output <= original_node_id < self.nodes_cnt

        self.active_hidden_nodes_cnt -= 1

        # _______________________________________

        prev_neighbor_node_id = self.prev_node[original_node_id]
        next_neighbor_node_id = self.next_node[original_node_id]

        # original order: prev_neighbor_node_id  ->  original_node_id  ->  next_neighbor_node_id
        # new order:      prev_neighbor_node_id             ->             next_neighbor_node_id

        self.prev_node[original_node_id] = None
        self.next_node[original_node_id] = None

        self.next_node[prev_neighbor_node_id] = next_neighbor_node_id
        self.prev_node[next_neighbor_node_id] = prev_neighbor_node_id

    # =================================================

    # def add_connection(self, src_node_id, dst_node_id):
    def enable_connection(self, src_node_id, dst_node_id):
        assert self.is_existing_edge(src_node_id, dst_node_id)
        assert not self.conn[src_node_id][dst_node_id][0]
        self.conn[src_node_id][dst_node_id][0] = True

    # def delete_connection(self, src_node_id, dst_node_id):
    def disable_connection(self, src_node_id, dst_node_id):
        assert self.is_existing_edge(src_node_id, dst_node_id)
        assert self.conn[src_node_id][dst_node_id][0]
        self.conn[src_node_id][dst_node_id][0] = False
        # self.conn[src_node_id][dst_node_id][1] = 0

    # =================================================

    def evaluate(self, input_data, desired_output_data, derivative_unnecessary=False):

        value = {}
        for node_id in self.next_nodes(0):
            if self.is_input_node(node_id):
                value[node_id] = input_data[node_id]
            else:
                weighted_sum = 0
                for prev_node_id in self.previous_nodes(self.prev_node[node_id]):
                    if not self.is_enabled_edge(prev_node_id, node_id): continue
                    if self.conn[prev_node_id][node_id][0]:
                        weighted_sum += self.conn[prev_node_id][node_id][1] * value[prev_node_id]
                if weighted_sum < -100: weighted_sum = -100  # to avoid math.exp(- weighted_sum) throws OverflowError  # FIXME
                # value[node_id] = self.activate(weighted_sum)
                value[node_id] = 1.0 / (1 + math.exp(- weighted_sum))

        # estimated_output_data = value[self.d_input : self.d_input + self.d_output]
        estimated_output_data = [value[output_node_id] for output_node_id in range(self.d_input, self.d_input + self.d_output)]

        estimated_loss = 0
        for output_node_id in range(self.d_input, self.d_input + self.d_output):
            estimated_loss += abs(value[output_node_id] - desired_output_data[output_node_id - self.d_input])

        if derivative_unnecessary:
            return estimated_output_data, estimated_loss, {}, value, {}
        
        partial_derivative = {}
        for node_i_id in self.next_nodes(0):
            partial_derivative[node_i_id] = {}
            for node_j_id in self.next_nodes(node_i_id, including=False):
                # if node_j_id not in self.conn[node_i_id]: continue
                # if not self.conn[node_i_id][node_j_id][0]: continue
                # if not self.is_enabled_edge(node_i_id, node_j_id): continue  # The disabled connections are derivatived as if their weights are zeros.
                partial_derivative[node_i_id][node_j_id] = {}

                for node_k_id in self.previous_nodes(node_j_id, including=False):
                    partial_derivative[node_i_id][node_j_id][node_k_id] = 0

                partial_derivative[node_i_id][node_j_id][node_j_id] = value[node_j_id] * (1 - value[node_j_id]) * value[node_i_id]

                for node_k_id in self.next_nodes(node_j_id, including=False):
                    partial_derivative[node_i_id][node_j_id][node_k_id] = 0
                    for node_k0_id in self.previous_nodes(node_k_id, including=False):
                        if not self.is_enabled_edge(node_k0_id, node_k_id): continue
                        partial_derivative[node_i_id][node_j_id][node_k_id] += self.conn[node_k0_id][node_k_id][1] * partial_derivative[node_i_id][node_j_id][node_k0_id]
                    partial_derivative[node_i_id][node_j_id][node_k_id] *= value[node_k_id] * (1 - value[node_k_id])

        final_partial_derivative = {}
        for node_i_id in self.next_nodes(0):
            final_partial_derivative[node_i_id] = {}
            for node_j_id in self.next_nodes(node_i_id, including=False):
                # if not self.is_enabled_edge(node_i_id, node_j_id): continue  # The disabled connections are derivatived as if their weights are zeros.
                final_partial_derivative[node_i_id][node_j_id] = 0

                for output_node_id in range(self.d_input, self.d_input + self.d_output):
                    # final_partial_derivative[node_i_id][node_j_id] += partial_derivative[node_i_id][node_j_id][output_node_id] * (-1 if value[output_node_id] < desired_output_data[output_node_id - self.d_input] else 1)
                    final_partial_derivative[node_i_id][node_j_id] += partial_derivative[node_i_id][node_j_id][output_node_id] * 2 * (value[output_node_id]  -  desired_output_data[output_node_id - self.d_input])  # squared loss

        return estimated_output_data, estimated_loss, final_partial_derivative, value, partial_derivative

    # =================================================

    def evaluate_all(self, dataset, derivative_unnecessary=False):

        overall_loss = 0
        overall_final_partial_derivative = {
            node_i_id: {
                node_j_id: 
                    0.0
                for node_j_id in self.next_nodes(node_i_id, including=False)
                if self.is_enabled_edge(node_i_id, node_j_id)
            }
            for node_i_id in self.next_nodes(0)
        }
        for datapoint in dataset:
            _, loss, fpd, _, _ = self.evaluate(datapoint['input'], datapoint['output'], derivative_unnecessary)

            overall_loss += loss

            if not derivative_unnecessary:

                for node_i_id in self.next_nodes(0):
                    for node_j_id in self.next_nodes(node_i_id, including=False):
                        if not self.is_enabled_edge(node_i_id, node_j_id): continue
                        overall_final_partial_derivative[node_i_id][node_j_id] += fpd[node_i_id][node_j_id]

        return overall_loss, overall_final_partial_derivative

    # =================================================
    
    def __deepcopy__(self, memo=None):
        copied = DynamicGMP(d_input=self.d_input, d_output=self.d_output, lb_hidden=self.lb_hidden, ub_hidden=self.ub_hidden, is_deepcopying=True)
        copied.nodes_cnt = self.nodes_cnt
        copied.active_hidden_nodes_cnt = self.active_hidden_nodes_cnt
        copied.prev_node = copy.deepcopy(self.prev_node)
        copied.next_node = copy.deepcopy(self.next_node)
        copied.conn = copy.deepcopy(self.conn)
        return copied

    # =================================================

    def display(self):

        print('\n\n')

        print('\t', end='')
        for dst_node_id in self.next_nodes(0):
            if self.is_input_node(dst_node_id):
                continue

            style = '1;34' if self.is_hidden_node(dst_node_id) else '0'
            print(' \033[{}m#{:02d}\033[0m\t'.format(style, dst_node_id), end='')
        print()

        for src_node_id in self.next_nodes(0):
            print()

            style = '1;34' if self.is_hidden_node(src_node_id) else '0'
            print('\033[{}m#{:02d}\033[0m\t'.format(style, src_node_id), end='')

            for dst_node_id in self.next_nodes(0):
                if self.is_input_node(dst_node_id):
                    continue

                if src_node_id == dst_node_id:
                    print('\033[1;31mXXXX\033[0m\t', end='')
                elif not self.is_existing_edge(src_node_id, dst_node_id):
                    print('\t', end='')
                elif not self.is_enabled_edge(src_node_id, dst_node_id):
                    print('----\t', end='')
                else:
                    print('{:.2f}\t'.format(self.conn[src_node_id][dst_node_id][1]), end='')
            print()



class Mutation:

    def __init__(self):
        pass

    def operate(self, gmp:DynamicGMP):  # in-place operate, which modifies the original instance.
        raise NotImplementedError


class MutationNodeSplitting(Mutation):

    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha

    def operate(self, gmp:DynamicGMP, original_node_id, alpha=None):

        if alpha is None:
            alpha = self.alpha

        # _______________________________________

        prev_neighbor_node_id = gmp.prev_node[original_node_id]
        next_neighbor_node_id = gmp.next_node[original_node_id]

        # original order: prev_neighbor_node_id  ->  original_node_id  ->                                    ->  next_neighbor_node_id
        # new order:      prev_neighbor_node_id  ->                    ->  new_node_A_id  ->  new_node_B_id  ->  next_neighbor_node_id

        new_node_A_id = gmp.add_hidden_node(original_node_id, next_neighbor_node_id)
        new_node_B_id = gmp.add_hidden_node(new_node_A_id, next_neighbor_node_id)
        gmp.delete_hidden_node(original_node_id)
        # print('splitted node: {} -> {}(x) -> {}(new) -> {}(new) -> {}'.format(prev_neighbor_node_id, original_node_id, new_node_A_id, new_node_B_id, next_neighbor_node_id))

        # _______________________________________

        for node_id in gmp.previous_nodes(prev_neighbor_node_id):
            gmp.conn[node_id][new_node_A_id] = copy.deepcopy(gmp.conn[node_id][original_node_id])
            gmp.conn[node_id][new_node_B_id] = copy.deepcopy(gmp.conn[node_id][original_node_id])

        for node_id in gmp.next_nodes(next_neighbor_node_id):
            gmp.conn[new_node_A_id][node_id] = copy.deepcopy(gmp.conn[original_node_id][node_id])
            gmp.conn[new_node_B_id][node_id] = copy.deepcopy(gmp.conn[original_node_id][node_id])
            gmp.conn[new_node_A_id][node_id][1] *= (1 + alpha)
            gmp.conn[new_node_B_id][node_id][1] *= (-alpha)

        # gmp.conn[new_node_A_id][new_node_B_id] = (False, 0.0)


class MutationModifiedBackPropagation(Mutation):

    def __init__(
            self,
            init_learning_rate=0.5,
            # learning_rate_change=0.05,
            learning_rate_increase_multiple=1.05,
            learning_rate_decrease_multiple=0.6,
            lb_learning_rate=0.1,
            ub_learning_rate=0.6,
            learning_rate_adapt_epochs=10,  # k
            total_epochs=100,
        ):
        super().__init__()

        self.init_learning_rate = init_learning_rate
        # self.learning_rate_change = learning_rate_change  # ?
        self.learning_rate_increase_multiple = learning_rate_increase_multiple
        self.learning_rate_decrease_multiple = learning_rate_decrease_multiple
        self.lb_learning_rate = lb_learning_rate
        self.ub_learning_rate = ub_learning_rate

        self.learning_rate_adapt_epochs = learning_rate_adapt_epochs  # ?
        self.total_epochs = total_epochs  # ?

    def train(self, gmp:DynamicGMP, dataset, learning_rate):
            # in-place training
            # self.train(gmp, dataset, learning_rate)
            _, overall_final_partial_derivative = gmp.evaluate_all(dataset)

            overall_final_partial_derivative_vector_length = math.sqrt(sum([
                overall_final_partial_derivative[src_node_id][dst_node_id] ** 2
                for src_node_id in gmp.next_nodes(0)
                for dst_node_id in gmp.next_nodes(src_node_id, including=False)
                if gmp.is_enabled_edge(src_node_id, dst_node_id)
            ]))
            # print(overall_final_partial_derivative_vector_length)

            for src_node_id in gmp.next_nodes(0):
                for dst_node_id in gmp.next_nodes(src_node_id, including=False):
                    if not gmp.is_enabled_edge(src_node_id, dst_node_id): continue
                    gmp.conn[src_node_id][dst_node_id][1] -= learning_rate * overall_final_partial_derivative[src_node_id][dst_node_id] / overall_final_partial_derivative_vector_length

    def operate(self, gmp:DynamicGMP, dataset):

        last_overall_loss, _ = gmp.evaluate_all(dataset, derivative_unnecessary=True)
        last_gmp = copy.deepcopy(gmp)

        learning_rate = self.init_learning_rate

        for epoch_idx in range(1, self.total_epochs + 1):
            self.train(gmp, dataset, learning_rate)

            if epoch_idx % self.learning_rate_adapt_epochs == 0:
                current_overall_loss, _ = gmp.evaluate_all(dataset, derivative_unnecessary=True)
                print('\t\tAfter {:03d} epochs : learning_rate={:.3f} current_overall_loss={:.6f}'.format(epoch_idx, learning_rate, current_overall_loss))
                if current_overall_loss < last_overall_loss:
                    # learning_rate += self.learning_rate_change
                    learning_rate *= self.learning_rate_increase_multiple
                    if learning_rate >= self.ub_learning_rate: learning_rate = self.ub_learning_rate
                    
                    last_overall_loss = current_overall_loss
                    last_gmp = copy.deepcopy(gmp)
                else:
                    # learning_rate -= self.learning_rate_change
                    learning_rate *= self.learning_rate_decrease_multiple
                    if learning_rate <= self.lb_learning_rate: learning_rate = self.lb_learning_rate
                    
                    gmp = copy.deepcopy(last_gmp)
        print('\t\tFinished {} epochs.\n\n'.format(self.total_epochs))

    # def train(self, gmp:DynamicGMP, dataset, learning_rate):  # in-place modification


# class TrochModule(torch.nn.Module):

#     def __init__(self, gmp: DynamicGMP):
#         super().__init__()
#         self.gmp = gmp

#     def forward(self, x):
#         # value
#         # for node_id in self.gmp.next_nodes(0):
#         #     if self.gmp.is_input_node(node_id):
#         #         pass
#         #     elif self.gmp.is_hidden_node(node_id):
#         #         value[]
#         #     elif self.gmp.is_output_node(node_id):
#         estimated_output_data, _, _, _, _ = self.gmp.evaluate(x, [0 for _ in range(0, self.gmp.d_output)], derivative_unnecessary=True)
#         return estimated_output_data


# class MutationModifiedBackPropagationPyTorch(MutationModifiedBackPropagation):

#     def train(self, gmp: DynamicGMP, dataset, learning_rate):
#         gmp_troch = TrochModule(gmp)
#         gmp_troch.
#         gmp_troch.train()


class MutationModifiedBackPropagation_Focus(MutationModifiedBackPropagation):

    def __init__(self, init_learning_rate=0.5, learning_rate_change=0.05, lb_learning_rate=0.1, ub_learning_rate=0.6, learning_rate_adapt_epochs=10, total_epochs=100):
        super().__init__(init_learning_rate, learning_rate_change, lb_learning_rate, ub_learning_rate, learning_rate_adapt_epochs, total_epochs)
        print('\n\033[1;33mNote: using MutationModifiedBackPropagation_Focus.\033[0m')

    def train(self, gmp: DynamicGMP, dataset, learning_rate):

        datapoint_weights_sum = 0
        weighted_final_partial_derivative = {
            node_i_id: {
                node_j_id: 
                    0.0
                for node_j_id in gmp.next_nodes(node_i_id, including=False)
                if gmp.is_enabled_edge(node_i_id, node_j_id)
            }
            for node_i_id in gmp.next_nodes(0)
        }

        # FIXME  normalize vector ??

        for datapoint in dataset:
            _, loss, fpd, _, _ = gmp.evaluate(datapoint['input'], datapoint['output'], derivative_unnecessary=False)

            datapoint_weight = loss
            datapoint_weights_sum += datapoint_weight

            for node_i_id in gmp.next_nodes(0):
                for node_j_id in gmp.next_nodes(node_i_id, including=False):
                    if not gmp.is_enabled_edge(node_i_id, node_j_id): continue
                    weighted_final_partial_derivative[node_i_id][node_j_id] += fpd[node_i_id][node_j_id] * datapoint_weight

        datapoint_weights_avg = datapoint_weights_sum / len(dataset)

        for src_node_id in gmp.next_nodes(0):
            for dst_node_id in gmp.next_nodes(src_node_id, including=False):
                if not gmp.is_enabled_edge(src_node_id, dst_node_id): continue
                gmp.conn[src_node_id][dst_node_id][1] -= learning_rate * weighted_final_partial_derivative[src_node_id][dst_node_id] / datapoint_weights_avg


class MutationSimulatedAnnealing(Mutation):

    def __init__(
            self,
            fitness_function,
            temperatures_list,
            iterations_per_temperature,
        ):
        super().__init__()
        self.fitness_function = fitness_function
        self.temperatures_list = temperatures_list
        self.iterations_per_temperature = iterations_per_temperature

    def modify(self, modified_gmp:DynamicGMP):
        for (src_node_id, dst_node_id) in modified_gmp.all_edges(enabled_only=True):
            modified_gmp.conn[src_node_id][dst_node_id][1] *= random.random() * 0.2 + 0.9  # FIXME

    def operate(self, gmp: DynamicGMP, dataset):  # NOT in-place operate
        fitness_value = self.fitness_function(gmp, dataset)

        for temperature in self.temperatures_list:
            for iteration in range(1, self.iterations_per_temperature+1):
                modified_gmp = copy.deepcopy(gmp)
                self.modify(modified_gmp)
                modified_fitness_value = self.fitness_function(modified_gmp, dataset)

                fitness_value_change = modified_fitness_value - fitness_value
                accept_probability = math.exp(- fitness_value_change / temperature)
                accepted = (random.random() < accept_probability)

                if accepted:
                    gmp = modified_gmp
                    fitness_value = modified_fitness_value

        return gmp
        # TODO


# class MutationHiddenNodesDeletion(Mutation):

#     def __init__(self):
#         super().__init__()



# class Mutators:

#     def __init__(self, mutation_list):
#         self.mutation_list = mutation_list

#     def operate(self, gmp:DynamicGMP):  # in-place operate
#         for mutation in self.mutation_list:
#             mutation.operate()

