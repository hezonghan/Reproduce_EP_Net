
import random
import timeit


# from ep_net import simple_rank_based_selection


# hit = {}
# for iteration in range(1, 10000+1):
#     selected_idx, worst_idx = simple_rank_based_selection([1, 2, 3, 4, 5])
#     # print(selected_idx, worst_idx)
#     if selected_idx not in hit: hit[selected_idx] = 0
#     hit[selected_idx] += 1

# print(hit)



def test_deep_copy_efficiency():

    import copy

    from gmp import DynamicGMP

    N = 500000

    gmp = DynamicGMP(
        d_input=3,
        d_output=1,
        lb_hidden=1,
        ub_hidden=10,
        init_hidden_nodes_cnt=2,
        init_conn_density=0.75,
        init_weight_abs_ub=10.0,
    )

    t1 = timeit.default_timer()
    arr = [copy.deepcopy(gmp) for _ in range(0, N)]
    t2 = timeit.default_timer()
    print('{:.3f} s / {} instances'.format(t2-t1, N))

    # using copy.deepcopy for raw types:
    # 0.044 s / 1000 instances
    # 5.467 s / 100000 instances
    # 27.170 s / 500000 instances


    # manually deep copy:
    # 3.212 s / 100000 instances
    # 0.024 s / 1000 instances
    # 16.229 s / 500000 instances


if __name__ == '__main__':
    test_deep_copy_efficiency()
