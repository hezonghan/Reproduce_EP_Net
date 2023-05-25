
from ep_net import simple_rank_based_selection


hit = {}
for iteration in range(1, 10000+1):
    selected_idx, worst_idx = simple_rank_based_selection([1, 2, 3, 4, 5])
    # print(selected_idx, worst_idx)
    if selected_idx not in hit: hit[selected_idx] = 0
    hit[selected_idx] += 1

print(hit)
