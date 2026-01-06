import scipy.spatial
import torch


def to_set(edge_index):
    if torch.numel(edge_index) == 0:
        return edge_index
    return set([(i, j) for i, j in edge_index.t().tolist()])


def enforce_mnn(edge_index, mnn):
    if torch.numel(edge_index) == 0:
        return edge_index
    tmp = []
    count = 0
    current = -1
    for j, i in enumerate(edge_index[0, :]):
        if current == i:
            count += 1
        else:
            current = i
            count = 0
        if count < mnn:
            tmp.append(edge_index[:, j])
    return torch.vstack(tmp).T


def remove_loop(edge_index):
    return edge_index[:, edge_index[0, :] != edge_index[1, :]]


def reference_index(x, r, batch, loop, mnn):
    batchs_x = [x[batch == i] for i in torch.unique(batch)]
    tmp = []
    p_max = 0
    for b in batchs_x:
        tree = scipy.spatial.cKDTree(b.cpu().numpy())
        col = tree.query_ball_point(b.cpu(), r=r, return_sorted=True)
        truth = []
        for i, ns in enumerate(col):
            count = 0
            for j in ns:
                if ((i != j) or (loop)) and (count < mnn):
                    truth.append((i + p_max, j + p_max))
                    count += 1
        tmp.extend(truth)
        p_max += b.shape[0]
    return tmp


def time_pytorch_function(func, input):
    w = r = 1
    # Async CUDA timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Warmup
    for _ in range(w):
        func(*input)
    # Run
    start.record()
    for _ in range(r):
        func(*input)
    end.record()
    # Let CUDA kernels finish
    torch.cuda.synchronize()
    return start.elapsed_time(end)
