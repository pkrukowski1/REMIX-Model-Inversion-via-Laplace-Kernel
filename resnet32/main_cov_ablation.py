# -*-coding:utf8-*-
import argparse
import os
import runpy
import sys

from inversion.cov_models import COV_TYPES


def parse_cov_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cov_type', type=str, default='lcm', choices=COV_TYPES)
    parser.add_argument('--cov_rank', type=int, default=2,
                        help='lowrank: rank k (C(k+1) parameters; k=2 matches the LCM\'s 3C)')
    parser.add_argument('--cov_block', type=int, default=8,
                        help='blockdiag: block size b (C(b-1)/2 parameters; b=8 -> 3.5C)')
    parser.add_argument('--cov_edges_per_dim', type=float, default=2.0,
                        help='sparse: target number of precision edges / C (e=2 -> ~3C parameters)')
    parser.add_argument('--cov_shrinkage', type=str, default='oas',
                        help="shrinkage: 'oas' or a fixed intensity in [0, 1]")
    cov_args, rest = parser.parse_known_args(argv)
    if cov_args.cov_shrinkage != 'oas':
        cov_args.cov_shrinkage = float(cov_args.cov_shrinkage)
    return cov_args, rest


if __name__ == '__main__':
    cov_args, rest = parse_cov_args(sys.argv[1:])
    from continual_runner import cov_ablation
    cov_ablation.install(cov_args)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_task_contrastive_cl.py')
    sys.argv = [script] + rest
    runpy.run_path(script, run_name='__main__')
