
# class GMP:  # Generalized-Multilayer-Perceptrons

#     def __init__(self, d_input, d_output, ub_hidden):
#         self.d_input = d_input
#         self.d_output = d_output
#         self.ub_hidden = ub_hidden

#         self.nodes_order = None


# class FixedSizedGMP(GMP):

#     def __init__(self, d_input, d_output, ub_hidden):
#         super().__init__(d_input, d_output, ub_hidden)

#         # The "two matrices and one vector" implementation
#         # which is actually not convenient, e.g. during node splitting
#         self.hidden_node_enabled = [False for _ in range(0, ub_hidden)]
#         self.conn_enabled = [
#             [
#                 False
#                 for _ in range(0, d_output + ub_hidden)  # conn dst
#             ]
#             for _ in range(0, d_input + d_output + ub_hidden)  # conn src
#         ]
#         self.conn_weight = [
#             [
#                 0.0
#                 for _ in range(0, d_input + d_output + ub_hidden)
#             ]
#             for _ in range(0, d_input + d_output + ub_hidden)
#         ]


# class DynamicGMP(GMP):

#     def __init__(self, d_input, d_output, ub_hidden):
#         super().__init__(d_input, d_output, ub_hidden)

#         self.hidden_node_enabled = {hidden_node_id: False for hidden_node_id in range(0, ub_hidden)}



class DynamicGMP:

    def __init__(self) -> None:
        pass

        # self.prev_node = {
        #     node_id: None if (node_id == 1) else (node_id - 1)
        #     for node_id in range(1, self.nodes_cnt + 1)
        # }
        # self.next_node = {
        #     node_id: None if (node_id == self.nodes_cnt) else (node_id + 1)
        #     for node_id in range(1, self.nodes_cnt + 1)
        # }

    # =================================================

    def split_hidden_node(self, original_node_id, alpha=None):
        assert self.d_input + self.d_output <= original_node_id < self.nodes_cnt

        self.nodes_cnt += 2
        new_node_A_id = self.nodes_cnt - 2
        new_node_B_id = self.nodes_cnt - 1

        self.active_hidden_nodes_cnt += 1  # not 2

        # _______________________________________

        prev_neighbor_node_id = self.prev_node[original_node_id]
        next_neighbor_node_id = self.next_node[original_node_id]

        # original order: prev_neighbor_node_id  ->          original_node_id          ->  next_neighbor_node_id
        # new order:      prev_neighbor_node_id  ->  new_node_A_id  ->  new_node_B_id  ->  next_neighbor_node_id

        self.prev_node[original_node_id] = None
        self.next_node[original_node_id] = None

        self.next_node[prev_neighbor_node_id] = new_node_A_id
        self.next_node[new_node_A_id] = new_node_B_id
        self.next_node[new_node_B_id] = next_neighbor_node_id

        self.prev_node[next_neighbor_node_id] = new_node_B_id
        self.prev_node[new_node_B_id] = new_node_A_id
        self.prev_node[new_node_A_id] = prev_neighbor_node_id

        # _______________________________________

        for node_id in self.previous_nodes():
            self.conn[node_id][new_node_A_id] = copy.deepcopy(self.conn[node_id][original_node_id])
            self.conn[node_id][new_node_B_id] = copy.deepcopy(self.conn[node_id][original_node_id])

        if alpha is None:
            alpha = self.node_splitting_alpha
        self.conn[new_node_A_id] = {}
        self.conn[new_node_B_id] = {}
        for node_id in self.next_nodes():
            self.conn[new_node_A_id][node_id] = copy.deepcopy(self.conn[original_node_id][node_id])
            self.conn[new_node_B_id][node_id] = copy.deepcopy(self.conn[original_node_id][node_id])
            self.conn[new_node_A_id][node_id][1] *= (1 + alpha)
            self.conn[new_node_B_id][node_id][1] *= (-alpha)

        self.conn[new_node_A_id][new_node_B_id] = (False, 0.0)

    # def activate(self, s):  # v
    #     return 1.0 / (1 + math.exp(-s))
    #     # DO NOT OVERRIDE
    
    # def activate_derivative(self, s):
    #     return self.activate(s) * (1 - self.activate(s))

    def evaluate(self, input_data, desired_output_data):
        # partial_derivative = {
        #     src_node_id: {
        #         dst_node_id: 
        #             0.0
        #         for dst_node_id in self.next_nodes(self.next_node[src_node_id])
        #     }
        #     for src_node_id in self.next_nodes(0)
        # }
        pass



# class LearnableGMP(DynamicGMP):

#     def __init__(
#             self, 
#             d_input, d_output, ub_hidden, 
#             node_splitting_alpha=0.4, 
#             init_hidden_nodes_cnt=0, init_conn_density=0.75, init_weight_abs_ub=1,
#             init_learning_rate
#         ):
#         super().__init__(d_input, d_output, ub_hidden, node_splitting_alpha, init_hidden_nodes_cnt, init_conn_density, init_weight_abs_ub)

#     def modified_back_propagation(self):
        




