import yaml
import argparse
from trainer import train


# The YAML config is applied *after* argparse (args.update(param)), so it silently overwrites
# command-line values. These keys are re-applied from the command line when given explicitly,
# which is what the SLURM sweeps rely on (e.g. `--seed 1994` used to be ignored: every run used
# the YAML's `seed: [1993]`). --init_cls / --increment select another task split.
CLI_OVERRIDES = ('alpha_frob', 'seed', 'prefix', 'init_cls', 'increment')


def main():
    args = setup_parser().parse_args()

    cli_args = vars(args).copy()

    with open(args.config, 'r') as stream:
        try:
            param = yaml.safe_load(stream)
        except yaml.YAMLError:
            exit(0)

    args = vars(args)
    args.update(param)

    for key in CLI_OVERRIDES:
        if cli_args.get(key) is not None:
            args[key] = cli_args[key]
    for key in ('init_cls', 'increment'):  # previous argparse defaults, for configs without these keys
        if args.get(key) is None:
            args[key] = 10

    # trainer.train() iterates over a list of seeds; keep a single CLI seed as a one-element list.
    if not isinstance(args.get('seed'), list):
        args['seed'] = [args['seed']]
    if args.get('prefix') is None:
        args['prefix'] = ''

    apply_upper_bound(args, cli_args.get('upper_bound'))

    train(args)


def apply_upper_bound(args, mode):
    """Upper bounds for the MoE-Adapter experiments.

    joint       -- train once on all classes (a single task): no forgetting, no replay needed.
    real_replay -- identical pipeline and memory budget (`memory_per_class` per class), but the replay
                   buffer holds *real* training samples (in the same representation as the inverted
                   ones, i.e. the input of block `start_block`) instead of model-inversion samples.
                   This is the ceiling for any inversion method under the same budget.
    """
    args.setdefault('replay_mode', 'inversion')
    args['upper_bound'] = mode if mode is not None else 'none'
    if args['upper_bound'] == 'none':
        return
    args['alpha_frob'] = 0.0  # the LCM term plays no role in either bound
    if mode == 'joint':
        args['init_cls'] = args['n_classes']
        args['increment'] = args['n_classes']
    elif mode == 'real_replay':
        args['replay_mode'] = 'real'
    else:
        raise ValueError('Unknown upper bound: {}'.format(mode))


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='',
                        help='Yaml file of settings.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Overrides the seed list in the YAML (also sets the class order when shuffle=true).')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--tuned_epoch', type=int, default=5)
    parser.add_argument('--memory_per_class', type=int, default=20)
    parser.add_argument('--init_cls', type=int, default=None,
                        help='Overrides the YAML (classes in the first task).')
    parser.add_argument('--increment', type=int, default=None,
                        help='Overrides the YAML (classes per later task).')
    parser.add_argument('--pretrained_weight', type=str, default='laion400m_e32')
    parser.add_argument('--pool_size', type=int, default=100)
    parser.add_argument('--prompt_depth_text', type=int, default=12)
    parser.add_argument('--prompt_depth_vision', type=int, default=12)
    parser.add_argument('--n_ctx_vision', type=int, default=4)
    parser.add_argument('--n_ctx_text', type=int, default=4)
    parser.add_argument('--alpha_div', type=float, default=1)
    parser.add_argument('--gadi', action='store_true', default=False,
                        help='config on gadi machine')
    parser.add_argument('--local_path', type=str)
    parser.add_argument('--alpha_frob', type=float)
    parser.add_argument('--prefix', type=str, default=None,
                        help='Tag added to the log file name (keeps logs of different methods apart).')
    parser.add_argument('--upper_bound', type=str, default=None, choices=['none', 'joint', 'real_replay'],
                        help='Run an upper bound instead of the inversion-based method.')
    parser.add_argument('--save_pseudo_samples', action='store_true', default=False, help="Save generated images to disk")

    return parser


if __name__ == '__main__':
    main()
