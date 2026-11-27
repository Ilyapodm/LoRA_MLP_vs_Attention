"""The shared Trainer: sample-weighted loss, deterministic sampling, timings and AMP audit."""
import time

import torch
from transformers import Trainer, TrainerCallback

from data import EpochSampler


class NumericalInvalid(RuntimeError):
    pass


def image_loss(outputs, labels, num_items_in_batch=None):
    """Sum over this microbatch / images in the entire accumulation group."""
    logits = outputs['logits']
    if not torch.isfinite(logits).all().item():
        raise NumericalInvalid('Nonfinite forward logits')
    count = labels.numel() if num_items_in_batch is None else num_items_in_batch
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction='sum') / count
    if not torch.isfinite(loss).item():
        raise NumericalInvalid('Nonfinite forward loss')
    return loss


def classification_metrics(prediction):
    import numpy as np
    labels = prediction.label_ids
    correct = int(np.count_nonzero(prediction.predictions == labels))
    return {'correct_count': correct, 'sample_count': len(labels), 'accuracy': correct/len(labels)}


def argmax_logits(logits, labels):
    return logits.argmax(dim=-1)


class ImageTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, compute_loss_func=image_loss, **kwargs)
        self.validation_seconds = self.checkpoint_seconds = 0
        self.model_accepts_loss_kwargs = False
        if self.accelerator.num_processes != 1 or self.args.n_gpu > 1:
            raise ValueError('One process and at most one GPU per run are required')

    def _get_train_sampler(self, train_dataset=None):
        source = self.train_dataset if train_dataset is None else train_dataset
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return EpochSampler(source, seed)

    def _synchronize(self):
        if self.args.device.type == 'cuda':
            torch.cuda.synchronize()

    def evaluate(self, *args, **kwargs):
        self._synchronize()
        start = time.perf_counter()
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            self._synchronize()
            self.validation_seconds += time.perf_counter()-start

    def _save_checkpoint(self, *args, **kwargs):
        self._synchronize()
        start = time.perf_counter()
        try:
            return super()._save_checkpoint(*args, **kwargs)
        finally:
            self._synchronize()
            self.checkpoint_seconds += time.perf_counter()-start


class UpdateAudit(TrainerCallback):
    def __init__(self):
        self.trainer = None
        self.events = []
        self.consecutive_skips = 0

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        accelerator = self.trainer.accelerator
        skipped = bool(accelerator.optimizer_step_was_skipped)
        self.consecutive_skips = self.consecutive_skips+1 if skipped else 0
        scaler = accelerator.scaler
        self.events.append({'step': state.global_step, 'epoch': state.epoch,
            'skipped': skipped, 'scale': scaler.get_scale() if scaler else None,
            'lr_after_update': [group['lr'] for group in optimizer.param_groups]})
        if self.consecutive_skips >= 10:
            raise RuntimeError('technical_failed: 10 consecutive AMP skipped updates')
        if not all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
            raise NumericalInvalid('Nonfinite trainable weights')
