"""Trainer technical checks and full-data local LR search. Final/test remain gated."""
import argparse
import json
from pathlib import Path
import time
import hashlib
import importlib.metadata
import platform
import uuid

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from data import EpochSampler
from checkpoints import (create_bundle, publish_bundle, restore_bundle, prune_bundles,
                         StorageSyncError, file_hash)


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
    return {'correct_count': correct, 'sample_count': len(labels),
            'accuracy': correct / len(labels)}


def argmax_logits(logits, labels):
    return logits.argmax(dim=-1)


class ImageTrainer(Trainer):
    def __init__(self, *args, bundle_root=None, storage_root=None, run_identity=None, **kwargs):
        super().__init__(*args, compute_loss_func=image_loss, **kwargs)
        self.validation_seconds = 0
        self.checkpoint_seconds = 0
        self.bundle_seconds = self.sync_seconds = 0
        self.bundle_root, self.storage_root = bundle_root, storage_root
        self.run_identity = run_identity
        self.update_audit = None
        self.last_bundle = self.last_published_bundle = None
        self.trace_probe = False
        self.probe_trace = []
        # PEFT's **kwargs does not mean the wrapped ViT accepts num_items_in_batch.
        self.model_accepts_loss_kwargs = False
        if self.accelerator.num_processes != 1 or self.args.n_gpu > 1:
            raise ValueError('One process and at most one GPU per run are required')

    def _get_train_sampler(self, train_dataset=None):
        source = self.train_dataset if train_dataset is None else train_dataset
        return EpochSampler(source, self.args.data_seed if self.args.data_seed is not None else self.args.seed)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self.trace_probe:
            return super().training_step(model, inputs, num_items_in_batch)

        def digest(items):
            h = hashlib.sha256()
            for name, tensor in items:
                h.update(name.encode())
                h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        states = [(name+'/'+str(key), value) for name, p in parameters
                  for key, value in sorted(self.optimizer.state.get(p, {}).items()) if torch.is_tensor(value)]
        event = {'step_before': self.state.global_step,
                 'inputs': digest(sorted(inputs.items())), 'parameters': digest(parameters),
                 'optimizer': digest(states),
                 'cpu_rng': digest([('rng', torch.get_rng_state())]),
                 'cuda_rng': digest([('rng', torch.cuda.get_rng_state())]) if self.args.device.type == 'cuda' else None}
        loss = super().training_step(model, inputs, num_items_in_batch)
        event['gradients'] = digest((n, p.grad) for n, p in parameters if p.grad is not None)
        self.probe_trace.append(event)
        return loss

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
            self.validation_seconds += time.perf_counter() - start

    def _save_checkpoint(self, *args, **kwargs):
        self._synchronize()
        start = time.perf_counter()
        try:
            super()._save_checkpoint(*args, **kwargs)
        finally:
            self._synchronize()
            self.checkpoint_seconds += time.perf_counter() - start
        if self.bundle_root is not None:
            checkpoint = Path(self.args.output_dir)/f'checkpoint-{self.state.global_step}'
            best = Path(self.state.best_model_checkpoint or checkpoint)
            audit = self.update_audit
            metadata = {'schema_version': 1, 'identity': self.run_identity,
                        'global_step': self.state.global_step, 'epoch': self.state.epoch,
                        'best_global_step': self.state.best_global_step,
                        'best_correct_count': self.state.best_metric,
                        'amp': self.accelerator.scaler is not None,
                        'audit': {'events': audit.events, 'consecutive_skips': audit.consecutive_skips} if audit else None,
                        'device': str(self.args.device),
                        'gpu': torch.cuda.get_device_name() if self.args.device.type == 'cuda' else None}
            start = time.perf_counter()
            target = Path(self.bundle_root)/f'epoch-{round(self.state.epoch):04d}-step-{self.state.global_step:06d}'
            self.last_bundle = create_bundle(checkpoint, best, target, metadata)
            self.bundle_seconds += time.perf_counter()-start
            if self.storage_root is not None:
                start = time.perf_counter()
                self.last_published_bundle = publish_bundle(self.last_bundle, self.storage_root)
                self.sync_seconds += time.perf_counter()-start
                prune_bundles(self.storage_root, self.run_identity)
            prune_bundles(self.bundle_root, self.run_identity)


class UpdateAudit(TrainerCallback):
    """Observe Trainer/Accelerate updates without replacing their training loop."""
    def __init__(self):
        self.trainer = None
        self.events = []
        self.consecutive_skips = 0

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        accelerator = self.trainer.accelerator
        skipped = bool(accelerator.optimizer_step_was_skipped)
        self.consecutive_skips = self.consecutive_skips + 1 if skipped else 0
        scaler = accelerator.scaler
        self.events.append({'step': state.global_step, 'epoch': state.epoch,
                            'skipped': skipped, 'scale': scaler.get_scale() if scaler else None,
                            'lr_after_update': [g['lr'] for g in optimizer.param_groups]})
        if self.consecutive_skips >= 10:
            raise RuntimeError('technical_failed: 10 consecutive AMP skipped updates')
        if not all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
            raise NumericalInvalid('Nonfinite trainable weights')


class PrefixDataset:
    """Technical subset which retains epoch-dependent augmentations."""
    def __init__(self, dataset, size):
        self.dataset, self.size = dataset, min(size, len(dataset))

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def __getitem__(self, index):
        return self.dataset[index]


class StopAfterEpoch(TrainerCallback):
    def __init__(self, epoch):
        self.epoch = epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch is not None and state.epoch >= self.epoch:
            control.should_training_stop = True


def technical_identity(config, model_revision, manifest_hash, probe=False):
    root = Path(__file__).resolve().parent
    hashes = {name: file_hash(root/name) for name in
              ('configs.py', 'data.py', 'train.py', 'checkpoints.py', 'requirements.txt')}
    packages = {name: importlib.metadata.version(name) for name in
                ('torch', 'torchvision', 'transformers', 'peft', 'accelerate', 'datasets', 'numpy', 'Pillow',
                 'huggingface-hub', 'safetensors')}
    return {'schema_version': 1, 'stage': 'technical', 'config': config,
            'model_revision': model_revision, 'manifest_hash': manifest_hash, 'seed': 42,
            'train_images': 20 if probe else 772, 'val_images': 8 if probe else 32,
            'epochs': 3 if probe else 2, 'microbatch': 8 if probe else 16,
            'accumulation': 2 if probe else 32, 'eval_batch': 8 if probe else 32,
            'optimizer': 'adamw_torch', 'lr': 1e-3, 'weight_decay': 0,
            'scheduler': 'linear', 'warmup': 0, 'precision': 'fp32-amp-fp16-sdpa-tf32off',
            'code_hash': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            'python': list(platform.python_version_tuple()[:2]), 'packages': packages}


def main():
    from transformers import AutoImageProcessor, set_seed
    from configs import CONFIGS, MODEL_ID, load_model
    from data import read_manifest, official_train, ManifestDataset, collate_fn

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('technical',), required=True)
    p.add_argument('--config', choices=tuple(CONFIGS), default='B-iso')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model-revision', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime-profile', type=Path)
    p.add_argument('--resume-bundle', type=Path)
    p.add_argument('--storage-root', type=Path, help='Existing persistent directory; no automatic Drive mount')
    p.add_argument('--probe', action='store_true', help='Three tiny technical epochs for checkpoint acceptance')
    p.add_argument('--stop-after-epoch', type=int, choices=(1, 2))
    p.add_argument('--force-best-first', action='store_true', help='Technical fixture: force metric ties to retain epoch 1')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; choose a fresh technical attempt directory')
    if not torch.cuda.is_available():
        p.error('This real ViT technical run requires CUDA; unit tests use small CPU models')
    if args.storage_root is not None and not args.storage_root.is_dir():
        p.error('Mount/create and check the persistent directory first')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    set_seed(42)
    manifest = read_manifest(args.manifest)
    identity = technical_identity(args.config, args.model_revision, manifest['content_hash'], args.probe)
    identity['force_best_first'] = args.force_best_first
    if args.runtime_profile:
        from experiment import validate_profile
        profile = validate_profile(json.loads(args.runtime_profile.read_text()))
        if args.probe: p.error('--probe cannot override the scientific runtime profile')
        identity.update({k: profile[k] for k in ('microbatch', 'accumulation', 'eval_batch', 'gradient_checkpointing')})
    source = official_train(manifest['dataset_revision'])
    if source.features['label'].names != manifest['label_names']:
        raise ValueError('Source labels do not match the common manifest')
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=args.model_revision, use_fast=False)
    # 512 + 260 images exercises the protocol's actual incomplete group size.
    train = PrefixDataset(ManifestDataset(source, manifest, 'train', processor, 42), identity['train_images'])
    val = PrefixDataset(ManifestDataset(source, manifest, 'val', processor, 42), identity['val_images'])
    model = load_model(args.config, 42, args.model_revision, manifest['label_names'])
    args.output.mkdir(parents=True, exist_ok=False)
    settings = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=identity['epochs'],
        per_device_train_batch_size=identity['microbatch'], gradient_accumulation_steps=identity['accumulation'],
        per_device_eval_batch_size=identity['eval_batch'], learning_rate=1e-3, weight_decay=0,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        lr_scheduler_type='linear', warmup_steps=0, max_grad_norm=1,
        optim='adamw_torch', fp16=True, tf32=False, seed=42, data_seed=42,
        gradient_checkpointing=identity.get('gradient_checkpointing', False) and args.config != 'LP',
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=0, dataloader_drop_last=False,
        remove_unused_columns=False, label_names=['labels'],
        eval_strategy='epoch', save_strategy='epoch', logging_steps=1,
        load_best_model_at_end=True, metric_for_best_model='correct_count',
        greater_is_better=True, save_total_limit=2, save_safetensors=True,
        report_to='none', skip_memory_metrics=True,
    )
    audit = UpdateAudit()
    resume = None
    if args.resume_bundle:
        resume, metadata = restore_bundle(args.resume_bundle, args.output/'restored', identity)
        audit.events = metadata['audit']['events']
        audit.consecutive_skips = metadata['audit']['consecutive_skips']

    def metrics(prediction):
        result = classification_metrics(prediction)
        if args.force_best_first:
            result['actual_correct_count_before_fixture'] = result['correct_count']
            result['correct_count'] = 0  # Acceptance fixture only, never a scientific metric.
            result['accuracy'] = 0.0
        return result

    trainer = ImageTrainer(model=model, args=settings, train_dataset=train, eval_dataset=val,
                           data_collator=collate_fn, processing_class=processor,
                           compute_metrics=metrics,
                           preprocess_logits_for_metrics=argmax_logits,
                           callbacks=[audit, StopAfterEpoch(args.stop_after_epoch)],
                           bundle_root=args.output/'bundles', run_identity=identity,
                           storage_root=(args.storage_root/(args.output.name+'-'+uuid.uuid4().hex[:12])) if args.storage_root else None)
    audit.trainer = trainer
    trainer.update_audit = audit
    trainer.trace_probe = args.probe
    (args.output/'technical-config.json').write_text(json.dumps({
        'stage': 'technical', 'scientific_result': False, 'config': args.config,
        'manifest_hash': manifest['content_hash'], 'model_revision': args.model_revision,
        'identity': identity, 'train_images': len(train), 'val_images': len(val),
        'epochs': identity['epochs']}, indent=2))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    except BaseException as error:
        (args.output/'status.json').write_text(json.dumps({
            'stage': 'technical',
            'status': ('numerical_invalid' if isinstance(error, NumericalInvalid) else
                       'interrupted' if isinstance(error, StorageSyncError) else 'technical_failed'),
            'reason': f'{type(error).__name__}: {error}'}, indent=2))
        raise
    finally:
        (args.output/'updates.json').write_text(json.dumps(audit.events, indent=2))
        if args.probe:
            (args.output/'probe-trace.json').write_text(json.dumps(trainer.probe_trace, indent=2))
    trainer.save_state()
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    (args.output/'technical-result.json').write_text(json.dumps({
        'stage': 'technical', 'scientific_result': False,
        'global_step': trainer.state.global_step, 'best_correct_count': trainer.state.best_metric,
        'best_checkpoint': trainer.state.best_model_checkpoint,
        'wall_train_job_seconds': wall, 'validation_seconds': trainer.validation_seconds,
        'checkpoint_seconds': trainer.checkpoint_seconds,
        'bundle_seconds': trainer.bundle_seconds, 'sync_seconds': trainer.sync_seconds,
        'train_loop_residual_seconds': wall-trainer.validation_seconds-trainer.checkpoint_seconds-trainer.bundle_seconds-trainer.sync_seconds,
        'last_bundle': str(trainer.last_bundle), 'published_bundle': str(trainer.last_published_bundle),
        'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
        'applied_steps': sum(not e['skipped'] for e in audit.events),
        'skipped_steps': sum(e['skipped'] for e in audit.events)}, indent=2))
    (args.output/'status.json').write_text(json.dumps({
        'stage': 'technical', 'status': 'interrupted' if trainer.state.epoch < identity['epochs'] else 'completed',
        'reason': 'requested_technical_stop' if trainer.state.epoch < identity['epochs'] else 'Short technical run only; no LR search/final'}, indent=2))


def export_best(checkpoint, destination, identity, best_epoch, correct_count):
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file
    destination = Path(destination);destination.mkdir(exist_ok=False)
    checkpoint = Path(checkpoint)
    if identity['run']['config'] == 'LP':
        with safe_open(str(checkpoint/'model.safetensors'), framework='pt', device='cpu') as f:
            tensors = {key: f.get_tensor(key).float().contiguous() for key in ('classifier.weight', 'classifier.bias')}
        save_file(tensors, str(destination/'head.safetensors'))
    else:
        for name in ('adapter_model.safetensors', 'adapter_config.json'):
            shutil.copy2(checkpoint/name, destination/name)
    (destination/'metadata.json').write_text(json.dumps({**identity, 'best_epoch': best_epoch,
        'val_correct_count': correct_count, 'val_sample_count': 2020}, indent=2))


def execute_lr_search(model, train_dataset, val_dataset, settings, output, config_record, env,
                      collator, processor=None):
    """Single scientific job; injectable model/data permit CPU integration tests."""
    from experiment import finish_result
    from checkpoints import write_json
    output = Path(output);output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', config_record)
    write_json(output/'environment.json', env)
    audit = UpdateAudit()
    trainer = ImageTrainer(model=model, args=settings, train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=collator, processing_class=processor, compute_metrics=classification_metrics,
        preprocess_logits_for_metrics=argmax_logits, callbacks=[audit])
    audit.trainer = trainer
    write_json(output/'status.json', {'stage': 'lr_search', 'status': 'training', 'reason': None})
    cuda = settings.device.type == 'cuda'
    if cuda: torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start = time.perf_counter()
    status, reason, error = 'valid', None, None
    metrics = {'test_accuracy': None, 'test_correct_count': None, 'test_sample_count': None,
               'epochs_completed': 0, 'val_correct_count': None, 'val_sample_count': None,
               'val_accuracy': None, 'best_epoch': None}
    try:
        trainer.train()
        if trainer.state.epoch != 5:
            raise RuntimeError('Scientific run must complete exactly five epochs')
        evaluations = [h for h in trainer.state.log_history if 'eval_correct_count' in h]
        if len(evaluations) != 5 or any(h['eval_sample_count'] != 2020 for h in evaluations):
            raise RuntimeError('Expected five full validation evaluations')
        best = max(evaluations, key=lambda h: (h['eval_correct_count'], -h['epoch']))
        count, epoch = int(best['eval_correct_count']), int(best['epoch'])
        export_best(trainer.state.best_model_checkpoint, output/'best', config_record, epoch, count)
        metrics.update(epochs_completed=5, val_correct_count=count, val_sample_count=2020,
                       val_accuracy=count/2020, best_epoch=epoch, global_step=trainer.state.global_step)
    except NumericalInvalid as exc:
        status, reason = 'numerical_invalid', str(exc)
    except BaseException as exc:
        status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'technical_failed'
        reason, error = f'{type(exc).__name__}: {exc}', exc
    finally:
        if cuda: torch.cuda.synchronize()
        wall = time.perf_counter()-start
        metrics.update(epochs_completed=int(trainer.state.epoch or 0),
                       wall_train_job_seconds=wall, validation_seconds=trainer.validation_seconds,
                       checkpoint_seconds=trainer.checkpoint_seconds,
                       train_loop_residual_seconds=wall-trainer.validation_seconds-trainer.checkpoint_seconds,
                       applied_steps=sum(not e['skipped'] for e in audit.events),
                       skipped_steps=sum(e['skipped'] for e in audit.events),
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None)
        import math
        def clean(value):
            if isinstance(value, float) and not math.isfinite(value): return None
            if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list): return [clean(v) for v in value]
            return value
        write_json(output/'history.json', clean(trainer.state.log_history))
        write_json(output/'updates.json', audit.events)
        write_json(output/'metrics.json', metrics)
        # status.json is the only deliberately mutable file before COMPLETE.
        (output/'status.json').write_text(json.dumps({'stage': 'lr_search', 'status': status, 'reason': reason}, indent=2))
    if error is not None: raise error
    finish_result(output)
    return status


def lr_search_main():
    from transformers import AutoImageProcessor, set_seed
    from configs import load_model, MODEL_ID
    from data import official_train, read_manifest, ManifestDataset, collate_fn
    from experiment import read_plan
    from run_queue import environment, validate_admission
    p = argparse.ArgumentParser(description='Full Food-101 LR trial; test evaluation and resume are disabled')
    p.add_argument('--stage', choices=('lr_search',), required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--executor', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--admission', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Output exists; every retry needs a new attempt directory')
    plan = read_plan(args.plan, args.protocol)
    candidates = [r for r in plan['runs'] if r['run_id'] == args.run_id and r['executor'] == args.executor]
    if len(candidates) != 1: p.error('Run is not assigned to this executor')
    run, common = candidates[0], plan['common']
    env = environment();validate_admission(args.admission, plan, args.executor, env)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    set_seed(run['seed'])
    manifest = read_manifest(args.protocol/'data_manifest.json')
    source = official_train(common['dataset_revision'])
    if source.features['label'].names != manifest['label_names']: raise ValueError('Label order differs')
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=common['model_revision'], use_fast=False)
    # Scientific stages always use the whole manifest, never PrefixDataset or probe fixtures.
    train = ManifestDataset(source, manifest, 'train', processor, run['seed'])
    val = ManifestDataset(source, manifest, 'val', processor, run['seed'])
    if len(train) != 18180 or len(val) != 2020: raise ValueError('Wrong scientific split sizes')
    model = load_model(run['config'], run['seed'], common['model_revision'], manifest['label_names'])
    profile = common['profile']
    settings = TrainingArguments(output_dir=str(args.output/'checkpoints'), num_train_epochs=5,
        per_device_train_batch_size=profile['microbatch'], gradient_accumulation_steps=profile['accumulation'],
        per_device_eval_batch_size=profile['eval_batch'], learning_rate=run['lr'], weight_decay=0,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, lr_scheduler_type='linear', warmup_steps=0,
        max_grad_norm=1, optim='adamw_torch', fp16=True, tf32=False, seed=142, data_seed=142,
        dataloader_num_workers=0, dataloader_drop_last=False, remove_unused_columns=False, label_names=['labels'],
        gradient_checkpointing=profile['gradient_checkpointing'] and run['config'] != 'LP',
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='epoch', save_strategy='epoch', logging_steps=1, load_best_model_at_end=True,
        metric_for_best_model='correct_count', greater_is_better=True, save_total_limit=2,
        save_safetensors=True, report_to='none', skip_memory_metrics=True)
    # Trainer creates its output directory in __init__, after execute_lr_search creates the attempt.
    status = execute_lr_search(model, train, val, settings, args.output,
        {'common': common, 'run': run, 'plan_hash': plan['plan_hash']}, env, collate_fn, processor)
    if status == 'numerical_invalid': raise SystemExit(3)


if __name__ == '__main__':
    import sys
    if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1:][:1] == ['lr_search']:
        lr_search_main()
    else:
        main()
