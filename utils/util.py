import torch


def count_total_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def concat_config(args, total_params):
    name = [
        f'nc{args.nc}',
        f'nct{args.nc_test}',
        f'ni{args.ni}',
        f'nb{args.nb}',
        f'nl{args.n_layers}',
        f'is{args.in_size}',
        f'hs{args.hidden_size}',
        f'os{args.out_size}',
        f's{args.seed}',
        f'lr{args.lr}',
        f'e{args.epochs}',
        f'tp{total_params}'
    ]
    return '_'.join(name)


def print_model_config(model):
    print('Layer shape:')
    for n, p in model.named_parameters():
        if 'weight' in n:
            print(n, p.shape)
    print(f'Total params: {count_total_params(model)}')


def concat_features(in_size, hidden_size, out_size, n_layers):
    return [in_size] + [hidden_size] * (n_layers-1) + [out_size]


def squared_mean(x):
    return torch.mean(x**2)


def relative_l2(u, u_gt):
    return torch.linalg.norm(u-u_gt) / torch.linalg.norm(u_gt)
