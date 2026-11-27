"""V2 full-manifest LR/final workers using the accepted Trainer, with no test loader."""
import argparse
import json
from pathlib import Path
import time

from checkpoints import write_json
from study_protocol import (config_record, read_plan, validate_plan, validate_freeze,
                            steps_per_epoch)
from study_results import seal_result


def clean_json(value):
    """Diagnostic overflow is explicit null, never non-standard NaN/Infinity JSON."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def _arguments(common, run, output, epochs, use_cpu):
    from transformers import TrainingArguments
    p, r = common['profile'], common['recipe']
    return TrainingArguments(
        output_dir=str(Path(output)/'checkpoints'), num_train_epochs=epochs,
        per_device_train_batch_size=p['microbatch'], gradient_accumulation_steps=p['accumulation'],
        per_device_eval_batch_size=p['eval_batch'], learning_rate=run['lr'],
        weight_decay=r['weight_decay'], adam_beta1=r['betas'][0], adam_beta2=r['betas'][1],
        adam_epsilon=r['epsilon'], lr_scheduler_type=r['scheduler'], warmup_steps=r['warmup_steps'],
        max_grad_norm=r['max_grad_norm'], label_smoothing_factor=r['label_smoothing'],
        optim=r['optimizer'], fp16=not use_cpu, use_cpu=use_cpu, tf32=False,
        seed=run['seed'], data_seed=run['seed'],
        dataloader_num_workers=p['workers'], dataloader_drop_last=False,
        remove_unused_columns=False, label_names=['labels'],
        gradient_checkpointing=p['gradient_checkpointing'] and run['config'] != 'LP',
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='epoch', save_strategy='epoch', logging_steps=1,
        load_best_model_at_end=True, metric_for_best_model='correct_count',
        greater_is_better=True, save_total_limit=2, save_safetensors=True,
        report_to='none', skip_memory_metrics=True)


def training_arguments(plan, run, output, *, use_cpu=False):
    return _arguments(plan['common'], run, output, plan['common']['epochs'], use_cpu)


def configure_runtime(seed):
    import torch
    from transformers import set_seed
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    set_seed(seed)


def execute_training(model, train_dataset, val_dataset, settings, output, plan, run,
                     env, collator=None, processor=None, setup_seconds=0):
    """Injectable models/data support CPU tests; CLI always loads the real frozen recipe."""
    import torch
    from train import (ImageTrainer, UpdateAudit, NumericalInvalid, classification_metrics,
                       argmax_logits, export_best)
    validate_plan(plan)
    identity = config_record(plan, run)
    common, stage = plan['common'], plan['stage']
    if (len(train_dataset), len(val_dataset)) != (common['train_images'], common['val_images']):
        raise ValueError('Scientific v2 jobs require the full 18180/2020 manifest')
    expected = training_arguments(plan, run, output, use_cpu=settings.use_cpu)
    keys = ('output_dir', 'num_train_epochs', 'max_steps', 'per_device_train_batch_size',
            'gradient_accumulation_steps', 'per_device_eval_batch_size', 'learning_rate',
            'seed', 'data_seed', 'optim', 'weight_decay', 'adam_beta1', 'adam_beta2',
            'adam_epsilon', 'lr_scheduler_type', 'warmup_steps', 'warmup_ratio',
            'max_grad_norm', 'label_smoothing_factor', 'fp16', 'bf16', 'tf32',
            'dataloader_num_workers', 'dataloader_drop_last', 'gradient_checkpointing',
            'gradient_checkpointing_kwargs', 'eval_strategy', 'save_strategy',
            'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better',
            'save_total_limit', 'save_safetensors', 'remove_unused_columns', 'label_names')
    if any(getattr(settings, k) != getattr(expected, k) for k in keys):
        raise ValueError('Trainer settings differ from the immutable v2 recipe')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', identity)
    write_json(output/'environment.json', env)

    def status_record(status, reason=None):
        return {'schema_version': 2, 'stage': stage, 'action': 'train',
                'status': status, 'reason': reason}

    write_json(output/'status.json', status_record('training'))
    audit, trainer = UpdateAudit(), None
    status = 'valid' if stage == 'lr_search' else 'trained'
    reason = error = None
    start = wall = None
    cuda = settings.device.type == 'cuda'
    metrics = {'test_accuracy': None, 'test_correct_count': None, 'test_sample_count': None,
               'val_accuracy': None, 'val_correct_count': None, 'val_sample_count': None,
               'best_epoch': None, 'epochs_completed': 0, 'global_step': 0,
               'setup_seconds': setup_seconds, 'export_seconds': 0,
               'peak_allocated_gib': None, 'peak_reserved_gib': None}
    try:
        trainer = ImageTrainer(model=model, args=settings, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=collator, processing_class=processor,
            compute_metrics=classification_metrics, preprocess_logits_for_metrics=argmax_logits,
            callbacks=[audit])
        audit.trainer = trainer
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        trainer.train()
        if cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter()-start
        if (trainer.state.epoch != common['epochs'] or
                trainer.state.global_step != common['epochs']*steps_per_epoch(common)):
            raise RuntimeError('V2 must complete ten epochs / 360 attempted updates')
        evaluations = [h for h in trainer.state.log_history if 'eval_correct_count' in h]
        if len(evaluations) != common['epochs'] or any(h['eval_sample_count'] != 2020 for h in evaluations):
            raise RuntimeError('Expected ten full validation evaluations')
        best = max(evaluations, key=lambda h: (h['eval_correct_count'], -h['epoch']))
        epoch, count = int(best['epoch']), int(best['eval_correct_count'])
        if trainer.state.best_metric != count or trainer.state.best_global_step != epoch*steps_per_epoch(common):
            raise RuntimeError('Trainer best differs from maximum correct_count/earliest epoch')
        export_start = time.perf_counter()
        export_best(trainer.state.best_model_checkpoint, output/'best', identity, epoch, count)
        metrics.update(val_accuracy=count/2020, val_correct_count=count, val_sample_count=2020,
                       best_epoch=epoch, export_seconds=time.perf_counter()-export_start)
    except NumericalInvalid as exc:
        status, reason = 'numerical_invalid', str(exc)
    except BaseException as exc:
        status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'technical_failed'
        reason, error = f'{type(exc).__name__}: {exc}', exc
    finally:
        if cuda:
            try:
                torch.cuda.synchronize()
                metrics.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            except RuntimeError as exc:
                if error is None:
                    status, reason, error = 'technical_failed', f'CUDA finalization: {exc}', exc
        if start is not None and wall is None:
            wall = time.perf_counter()-start
        val_seconds = trainer.validation_seconds if trainer else 0
        save_seconds = trainer.checkpoint_seconds if trainer else 0
        metrics.update(epochs_completed=int(trainer.state.epoch or 0) if trainer else 0,
                       global_step=trainer.state.global_step if trainer else 0,
                       wall_train_job_seconds=wall or 0, validation_seconds=val_seconds,
                       checkpoint_seconds=save_seconds,
                       train_loop_residual_seconds=(wall or 0)-val_seconds-save_seconds,
                       attempted_steps=len(audit.events),
                       applied_steps=sum(not e['skipped'] for e in audit.events),
                       skipped_steps=sum(e['skipped'] for e in audit.events))

        write_json(output/'history.json', clean_json(trainer.state.log_history if trainer else []))
        write_json(output/'updates.json', clean_json(audit.events))
        write_json(output/'metrics.json', metrics)
        record = status_record(status, reason)
        if stage == 'final':
            write_json(output/'training_status.json', record)
        (output/'status.json').write_text(json.dumps(record, indent=2)+'\n')
    if error is not None:
        raise error
    seal_result(output, plan, run)
    return status


def main():
    from transformers import AutoImageProcessor
    from study_configs import MODEL_ID, load_model
    from data import read_manifest, official_train, ManifestDataset, collate_fn
    from study_queue import environment, validate_admission
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'protocol', 'admission', 'output'):
        p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--freeze', type=Path)
    p.add_argument('--executor', required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; retries require a fresh attempt')
    plan = read_plan(args.plan, args.protocol)
    if plan['stage'] == 'final':
        validate_freeze(args.freeze or args.protocol/'freeze-v2.json', plan)
    candidates = [r for r in plan['runs'] if r['run_id'] == args.run_id and r['executor'] == args.executor]
    if len(candidates) != 1:
        p.error('Task is not assigned to this executor')
    run, common = candidates[0], plan['common']
    env = environment()
    validate_admission(args.admission, plan, args.executor, env)
    configure_runtime(run['seed'])
    setup_start = time.perf_counter()
    manifest = read_manifest(args.protocol/'data_manifest.json')
    source = official_train(common['dataset_revision'])
    if source.features['label'].names != manifest['label_names']:
        raise ValueError('Dataset class order differs from the common manifest')
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=common['model_revision'], use_fast=False)
    train = ManifestDataset(source, manifest, 'train', processor, run['seed'])
    val = ManifestDataset(source, manifest, 'val', processor, run['seed'])
    model = load_model(run['config'], run['seed'], common['model_revision'], manifest['label_names'])
    settings = training_arguments(plan, run, args.output)
    status = execute_training(model, train, val, settings, args.output, plan, run, env,
                              collate_fn, processor, time.perf_counter()-setup_start)
    if status == 'numerical_invalid':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
