import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_cov_ablation import load_run, select_best  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='<BASE_PATH>/<dataset>_10task')
    parser.add_argument('--tags', nargs='+', required=True, help='candidate run directories, in grid order')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--metric', default='final', choices=['final', 'avg_inc'])
    args = parser.parse_args()

    values, missing = {}, []
    for tag in args.tags:
        run_dir = os.path.join(args.root, tag, 'seed_{}'.format(args.seed))
        if os.path.exists(os.path.join(run_dir, 'accs.pkl')):
            values[tag] = load_run(run_dir)[args.metric]
        else:
            missing.append(run_dir)
    if missing:
        sys.exit('Tuning runs not finished: {}'.format(', '.join(missing)))
    best = select_best(values, args.tags)
    for tag in args.tags:
        print('  {:<28s} {} = {:.2f}{}'.format(tag, args.metric, values[tag], '   <- selected' if tag == best else ''),
              file=sys.stderr)
    print(args.tags.index(best))


if __name__ == '__main__':
    main()
