import os
import json
import pdb
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from collections import defaultdict
import math

import numpy as np
import torch
from torch import nn
from torch import device, Tensor
from tqdm.autonotebook import trange
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import distributed as dist
import transformers

WEIGHTS_NAME = "pytorch_model.bin"

class Trainer:
    '''trainer for single-gpu training.'''
    def __init__(self, args=None):
        pass

    def train(
        self,
        model,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        eval_dataloader=None,
        evaluator=None,
        epochs: int = 1,
        steps_per_epoch=None,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 100,
        save_steps: int = 100,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        callback: Callable = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_total_limit: int = 0,
        load_best_model_at_last: bool = True,
    ):

        self.best_score = -9999999
        self.accumulation_steps = accumulation_steps
        self.score_logs = defaultdict(list)
        self.evaluator = evaluator
        self.eval_dataloader = eval_dataloader

        dataloaders = [dataloader for dataloader, _, _ in train_objectives]
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio)

        loss_models = [loss for _, loss, _ in train_objectives]
        train_weights = [weight for _, _, weight in train_objectives]

        # ✅ FIX DEVICE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizers = []
        schedulers = []

        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay': weight_decay
                },
                {
                    'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(
                optimizer,
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                t_total=num_train_steps
            )

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]
        train_loss_dict = defaultdict(list)

        # ✅ FIX AMP for new torch
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            for train_iter in trange(steps_per_epoch, desc="Iteration"):

                for train_idx in range(len(train_objectives)):

                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]
                    loss_weight = train_weights[train_idx]

                    loss_model.train()

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    # ✅ FIX DEVICE MOVE
                    data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}

                    optimizer.zero_grad()

                    if use_amp:
                        with torch.autocast(device_type="cuda"):
                            loss_model_return = loss_model(**data)

                        loss_value = loss_weight * loss_model_return['loss_value'] / self.accumulation_steps

                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss_model_return = loss_model(**data)
                        loss_value = loss_weight * loss_model_return['loss_value'] / self.accumulation_steps
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    train_loss_dict[train_idx].append(loss_value.item())

                scheduler.step()
                global_step += 1

                if evaluation_steps > 0 and global_step % evaluation_steps == 0:
                    print('\nTrain loss:')
                    for k in train_loss_dict:
                        print(k, np.mean(train_loss_dict[k]))
                    train_loss_dict = defaultdict(list)

                if self.evaluator is not None and global_step % evaluation_steps == 0:
                    scores = self.evaluator.evaluate()
                    print(scores)

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        scheduler = scheduler.lower()

        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(
                optimizer, warmup_steps, t_total
            )
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, t_total
            )
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, warmup_steps, t_total
            )
        else:
            raise ValueError(f"Unknown scheduler {scheduler}")