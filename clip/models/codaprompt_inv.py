import copy
import logging
import os
import pickle

import numpy as np
import torch
from torch.nn import functional as F

from models.codaprompt import Learner as CodaLearner
from inversion import layer_wise_clip_inversion_prompt
from inversion import utils as inv_utils
from inversion.functions import WelfordCovarianceTracker
from inversion.lcm import LCM


class ReplayBuffer(object):
    def __init__(self, samples, queries, labels):
        self.samples, self.queries, self.labels = samples, queries, labels

    def __len__(self):
        return self.labels.shape[0]

    def sample(self, size, device):
        idx = torch.randint(0, len(self), (size,))
        return self.samples[idx].to(device), self.queries[idx].to(device), self.labels[idx].to(device)


class Learner(CodaLearner):
    # CODA-Prompt with model-inversion replay: PMI when alpha_frob = 0, REMIX when alpha_frob > 0.
    def __init__(self, args):
        super().__init__(args)
        if args['start_block'] != 1:
            raise ValueError('codaprompt_inv replays patch-embedding outputs: set start_block: 1')
        self.alpha_frob = float(args.get('alpha_frob') or 0.0)
        self.local_path = args['local_path']
        self.buffer = None
        self.cls2stat = {}
        self.lcms = {}
        self.old_model = self._frozen_copy()
        self.inversion_runner = layer_wise_clip_inversion_prompt.LayerWiseCLIPInversion(
            local_path=os.path.join(self.local_path, 'buffer'),
            model=self.old_model.clip_model.visual,
            image_size=[int(si) for si in args['img_size'].split(',')],
            lr=args['inv_lr'],
            train_steps=args['train_steps'],
            alpha_pr=args['alpha_pr'],
            alpha_rf=args['alpha_rf'],
            scheduler_params=None,
            use_rf=True,
            smooth_type='tv',
            flip_rate=0,
            log_step=200,
            opt_type='adam',
            boost_factor=False,
            loss_type='mse',
            grad_norm=None,
            clip_input='clip',
            input_aug=bool(args['input_aug']),
            save_step=0,
            pre_size_change=None,
            normalize=bool(args['feat_norm']),
            rf_factor=args['rf_factor'],
            start_block=args['start_block'],
            alpha_frob=self.alpha_frob
        )
        if torch.cuda.is_available():
            self.inversion_runner.cuda()

    def _frozen_copy(self):
        model = copy.deepcopy(self._net()).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def incremental_train(self, data_manager, tb_logger=None):
        super().incremental_train(data_manager, tb_logger)
        if self._total_classes < data_manager.nb_classes:
            self._prepare_replay()

    def _batch_loss(self, inputs, targets):
        loss, logits = super()._batch_loss(inputs, targets)
        if self.buffer is None or self._known_classes == 0:
            return loss, logits
        x, q, y = self.buffer.sample(inputs.shape[0], self._device)
        out, _ = self._network(x, q=q, skip_conv=True, task_id=self._cur_task, train=True)
        out = out[:, :self._total_classes]
        loss_ce = F.cross_entropy(out, y)
        with torch.no_grad():
            teacher = self.old_model(x, q=q, skip_conv=True)[:, :self._known_classes]
        temp = float(self.args.get('kd_temp', 2.0))
        loss_kd = F.kl_div(F.log_softmax(out[:, :self._known_classes] / temp, dim=1),
                           F.softmax(teacher / temp, dim=1), reduction='batchmean') * temp * temp
        loss = loss + float(self.args.get('lambda_replay', 1.0)) * loss_ce + float(self.args.get('lambda_kd', 1.0)) * loss_kd
        return loss, logits

    def _prepare_replay(self):
        self.old_model = self._frozen_copy()
        self.inversion_runner.update_model(model=self.old_model.clip_model.visual)
        self.inversion_runner.update_input_stat(data_loader=self.train_loader, model=self.old_model)
        self._update_class_stats(self.train_loader)
        if self.alpha_frob > 0.0:
            self._fit_lcms(self.train_loader)
        self.buffer = self._build_inv_data()
        logging.info('Inverted replay buffer: {} samples for {} classes'.format(len(self.buffer), len(self.cls2stat)))

    def _update_class_stats(self, loader):
        sums = {}
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                q = self.old_model.query(inputs)
                feat = self.old_model(inputs, pen=True, q=q)
                for c in targets.unique().tolist():
                    m = (targets == c).to(self._device)
                    s = sums.setdefault(c, [0, 0.0, 0.0, 0.0, 0.0])
                    s[0] += int(m.sum())
                    s[1] = s[1] + feat[m].double().sum(0)
                    s[2] = s[2] + (feat[m].double() ** 2).sum(0)
                    s[3] = s[3] + q[m].double().sum(0)
                    s[4] = s[4] + (q[m].double() ** 2).sum(0)
        for c, (n, f1, f2, q1, q2) in sums.items():
            fm, qm = f1 / n, q1 / n
            fs = (f2 / n - fm ** 2).clamp_min(0).sqrt()
            qs = (q2 / n - qm ** 2).clamp_min(0).sqrt()
            self.cls2stat[c] = [t.float().cpu() for t in (fm, fs, qm, qs)]

    def _stat_names(self):
        buffer_dir = os.path.join(self.local_path, 'buffer')
        visual = self.old_model.clip_model.visual
        id2name = {id(m): n for n, m in visual.named_modules()}
        name2stat = {}
        main_file = os.path.join(buffer_dir, 'model_input_stats.pkl')
        if os.path.exists(main_file):
            with open(main_file, 'rb') as fr:
                name2stat.update(pickle.load(fr))
        blocks, _ = inv_utils.split_clip_blocks(model=visual, normalize=True, split_cnn=True)
        for bid in range(len(blocks)):
            b_file = os.path.join(buffer_dir, 'block_input_stats{}.pkl'.format(bid))
            if not os.path.exists(b_file):
                continue
            with open(b_file, 'rb') as fr:
                b_stats = pickle.load(fr)
            for local_name, mi in blocks[bid].named_modules():
                if local_name in b_stats and local_name != 'total' and id(mi) in id2name:
                    name2stat[id2name[id(mi)]] = b_stats[local_name]
        return name2stat

    def _fit_lcms(self, loader):
        visual = self.old_model.clip_model.visual
        names = set(self._stat_names())
        trackers = {}
        active = [False]

        def make_hook(name):
            def hook(module, inp, out):
                if active[0]:
                    x = inp[0].detach()
                    if name not in trackers:
                        trackers[name] = WelfordCovarianceTracker(num_features=x.shape[0], device=x.device)
                    trackers[name].update(x, batch_dim=1)
            return hook

        handles = [m.register_forward_hook(make_hook(n)) for n, m in visual.named_modules() if n in names]
        with torch.no_grad():
            for _, inputs, _ in loader:
                inputs = inputs.to(self._device)
                q = self.old_model.query(inputs)
                active[0] = True
                self.old_model.forward_feature(inputs, prompt=self.old_model.prompt, q=q, train=False)
                active[0] = False
        for h in handles:
            h.remove()

        n_new = self._total_classes - self._known_classes
        ratio_old = self._known_classes / self._total_classes
        for name, tracker in trackers.items():
            if tracker.n < 2:
                continue
            r_curr = tracker.get_correlation_matrix()
            if name not in self.lcms:
                self.lcms[name] = LCM(dim=r_curr.shape[0]).to(r_curr.device)
            lcm = self.lcms[name]
            if self._known_classes == 0:
                r_target = r_curr
            else:
                with torch.no_grad():
                    r_old = lcm.correlation().detach()
                r_target = ratio_old * r_old + (n_new / self._total_classes) * r_curr
            self._fit_lcm_correlation(lcm, r_target)
        self.inversion_runner.lcms = self.lcms
        logging.info('Fitted {} LCMs (class ratio of the new task {:.2f})'.format(len(trackers), n_new / self._total_classes))

    @staticmethod
    def _fit_lcm_correlation(lcm, target, epochs=200, lr=0.01):
        opt = torch.optim.Adam(lcm.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lcm.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.mse_loss(lcm.correlation(), target)
            loss.backward()
            opt.step()
            scheduler.step()
        lcm.eval()

    def _build_inv_data(self):
        args = self.args
        per_class = args['memory_per_class']
        feats, queries, labels = [], [], []
        for c in sorted(self.cls2stat):
            fm, fs, qm, qs = self.cls2stat[c]
            feats.append(fm + fs * torch.randn(per_class, fm.shape[0]))
            queries.append(qm + qs * torch.randn(per_class, qm.shape[0]))
            labels.append(torch.full((per_class,), c, dtype=torch.long))
        feats, queries, labels = torch.cat(feats), torch.cat(queries), torch.cat(labels)
        perm = torch.randperm(labels.shape[0])
        feats, queries, labels = feats[perm], queries[perm], labels[perm]

        size_change = [int(si) for si in args['size_change'].split(',')] if len(args['size_change']) > 0 else None
        if len(args['layer_size_change']) > 0:
            layer_iters = [int(si) for si in args['layer_size_change'].split(',')]
            first_layer_param = {'iters': layer_iters[-1], 'size_change': layer_iters[:-1], 'alpha_pr': args['l_pr'],
                                 'l_aug': bool(args['l_aug']), 'l_lr': args['l_lr']}
        else:
            first_layer_param = {'iters': None, 'size_change': None, 'alpha_pr': args['l_pr'],
                                 'l_aug': bool(args['l_aug']), 'l_lr': args['l_lr']}
        samples, count = [], 0
        while count < labels.shape[0]:
            bs = min(args['gen_batch_size'], labels.shape[0] - count)
            layer_batch = min(args['layer_batch'], labels.shape[0] - count)
            gen = self.inversion_runner.layer_wise_inversion_for_cl(
                batch_size=bs,
                target_feats=feats[count:count + layer_batch],
                query=queries[count:count + layer_batch],
                finetune_iters=args['tune_steps'],
                finetune_lr=args['tune_lr'],
                milestones=[int(mi) for mi in args['milestones'].split(',')],
                prompt=self.old_model.prompt,
                iters=args['train_steps'],
                size_change=size_change,
                return_best=bool(args['return_best']),
                first_layer_param=first_layer_param,
                search_param=bool(args['search_params']),
                verbose=False,
                gradual_rf=False
            )
            samples.append(gen.detach().cpu())
            count += gen.shape[0]
            logging.info('finish generating samples: {}'.format(count))
        return ReplayBuffer(torch.cat(samples), queries, labels)
