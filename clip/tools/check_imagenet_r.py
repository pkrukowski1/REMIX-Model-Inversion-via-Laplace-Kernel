import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import iImageNetR, load_json  # noqa: E402


def zero_shot(paths, labels, names, template, n, batch_size, device, seed):
    import torch
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    model.eval()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    results = {}
    with torch.no_grad():
        img_feats = []
        for s in range(0, len(idx), batch_size):
            batch = torch.stack([preprocess(Image.open(paths[i]).convert('RGB')) for i in idx[s:s + batch_size]])
            f = model.encode_image(batch.to(device)).float()
            img_feats.append(f / f.norm(dim=-1, keepdim=True))
        img_feats = torch.cat(img_feats)
        y = torch.as_tensor(labels[idx], device=device)
        for tag, cls_names in names.items():
            t = model.encode_text(tokenizer([template.format(c) for c in cls_names]).to(device)).float()
            t = t / t.norm(dim=-1, keepdim=True)
            pred = (img_feats @ t.T).argmax(dim=1)
            results[tag] = (pred == y).float().mean().item() * 100
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zeroshot', action='store_true')
    parser.add_argument('--n', type=int, default=2000, help='number of test images for the zero-shot check')
    parser.add_argument('--template', type=str, default='itap of a {}.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gadi', action='store_true')
    args = parser.parse_args()

    data = iImageNetR({'model_name': 'moe_adapter_inv', 'gadi': args.gadi})
    data.download_data()
    fixed = data.class_names
    old = list(load_json('utils/des_imagenetr.json').keys())[:200]

    print('train / test images: {} / {}'.format(len(data.train_data), len(data.test_data)))
    print('labels per split: {} / {}'.format(len(np.unique(data.train_targets)), len(np.unique(data.test_targets))))
    print('\nlabel  example image                           fixed name            old name')
    for c in [0, 1, 2, 50, 100, 150, 199]:
        example = data.test_data[np.where(data.test_targets == c)[0][0]]
        print('{:5d}  {:38s}  {:20s}  {}'.format(c, os.sep.join(example.split(os.sep)[-2:])[:38], fixed[c], old[c]))
    n_same = sum(a == b for a, b in zip(fixed, old))
    print('\n{} / 200 labels had the right name before the fix'.format(n_same))

    if args.zeroshot:
        acc = zero_shot(data.test_data, data.test_targets, {'fixed': fixed, 'old': old},
                        args.template, args.n, args.batch_size, args.device, seed=0)
        print('\nzero-shot CLIP ViT-B/16 on {} ImageNet-R test images ("{}"):'.format(args.n, args.template))
        for tag, a in acc.items():
            print('  {:5s} names: {:.2f}%'.format(tag, a))


if __name__ == '__main__':
    main()
