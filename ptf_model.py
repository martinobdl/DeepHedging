import torch


def ptf_function(paths, weights):

    # paths size is [batch_size, n_assets, time_steps]
    # weights size is [batch_size, n_assets-1, time_steps]

    old_w1 = torch.zeros_like(paths[:, :, 0])
    out = torch.zeros_like(paths)
    for i in range(paths.shape[2]-1):
        w1 = weights[:, :, i]
        W0 = torch.sum(old_w1*paths[:, :, i], dim=1).unsqueeze(1)
        w1_bond = (W0 - w1*paths[:, 1:, i])/paths[:, 0:1, i]
        w1 = torch.cat((w1_bond, w1), dim=1)
        old_w1 = w1
        out[:, :, i] = w1
    W0 = torch.sum(old_w1*paths[:, :, -1], dim=1).unsqueeze(1)
    out[:, 0, -1:] = W0/paths[:, 0:1, -1]
    return out
