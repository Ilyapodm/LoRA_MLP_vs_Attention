"""Train one frozen final task from pretrained weights; never construct a test loader."""
import argparse
import json
from pathlib import Path
import time

from checkpoints import write_json
from final_protocol import config_record, read_final_plan
from final_results import seal_training


def training_arguments(plan, run, output, *, use_cpu=False):
    """The CPU switch is for injected integration fixtures, not a scientific CLI option."""
    from transformers import TrainingArguments
    profile = plan['common']['profile']
    return TrainingArguments(
        output_dir=str(Path(output)/'checkpoints'), num_train_epochs=5,
        per_device_train_batch_size=profile['microbatch'],
        gradient_accumulation_steps=profile['accumulation'],
        per_device_eval_batch_size=profile['eval_batch'],
        learning_rate=run['lr'], weight_decay=0,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        lr_scheduler_type='linear', warmup_steps=0, max_grad_norm=1,
        optim='adamw_torch', fp16=not use_cpu, use_cpu=use_cpu, tf32=False,
        seed=run['seed'], data_seed=run['seed'],
        dataloader_num_workers=0, dataloader_drop_last=False,
        remove_unused_columns=False, label_names=['labels'],
        gradient_checkpointing=profile['gradient_checkpointing'] and run['config'] != 'LP',
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='epoch', save_strategy='epoch', logging_steps=1,
        load_best_model_at_end=True, metric_for_best_model='correct_count',
        greater_is_better=True, save_total_limit=2, save_safetensors=True,
        report_to='none', skip_memory_metrics=True)


def execute_final_training(model, train_dataset, val_dataset, settings, output,
                           plan, run, env, collator=None, processor=None):
    """Shared Trainer with injectable model/data; successful training stops at TRAINED."""
    import math
    import torch
    from train import (ImageTrainer, UpdateAudit, NumericalInvalid, classification_metrics,
                       argmax_logits, export_best)

    if (len(train_dataset), len(val_dataset)) != (18180, 2020):
        raise ValueError('Final requires the full scientific train/val sizes')
    expected_settings = training_arguments(plan, run, output, use_cpu=settings.use_cpu)
    # Test fixtures may change presentation settings, never the training budget/math.
    fields = ('num_train_epochs', 'max_steps', 'per_device_train_batch_size',
              'gradient_accumulation_steps', 'per_device_eval_batch_size', 'learning_rate',
              'seed', 'data_seed', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon',
              'lr_scheduler_type', 'warmup_steps', 'warmup_ratio', 'max_grad_norm', 'optim',
              'fp16', 'bf16', 'tf32', 'dataloader_num_workers', 'dataloader_drop_last',
              'gradient_checkpointing', 'gradient_checkpointing_kwargs', 'eval_strategy',
              'save_strategy', 'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better',
              'label_smoothing_factor', 'output_dir')
    if any(getattr(settings, k) != getattr(expected_settings, k) for k in fields):
        raise ValueError('Final Trainer settings differ from the frozen task')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = config_record(plan, run)
    write_json(output/'config.json', identity)
    write_json(output/'environment.json', env)

    def status_record(status, reason=None):
        return {'schema_version': 1, 'stage': 'final', 'action': 'train',
                'status': status, 'reason': reason}

    write_json(output/'status.json', status_record('training'))
    audit, trainer = UpdateAudit(), None
    metrics = {'test_accuracy': None, 'test_correct_count': None, 'test_sample_count': None,
               'val_accuracy': None, 'val_correct_count': None, 'val_sample_count': None,
               'best_epoch': None, 'epochs_completed': 0, 'global_step': 0,
               'wall_train_job_seconds': 0, 'export_seconds': 0,
               'peak_allocated_gib': None, 'peak_reserved_gib': None}
    status, reason, error = 'trained', None, None
    cuda = settings.device.type == 'cuda'
    start, wall = None, None
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
        if trainer.state.epoch != 5 or trainer.state.global_step != 180:
            raise RuntimeError('Final must complete five epochs and 180 attempted updates')
        evaluations = [h for h in trainer.state.log_history if 'eval_correct_count' in h]
        if len(evaluations) != 5 or any(h['eval_sample_count'] != 2020 for h in evaluations):
            raise RuntimeError('Expected five full validation evaluations')
        best = max(evaluations, key=lambda h: (h['eval_correct_count'], -h['epoch']))
        epoch, count = int(best['epoch']), int(best['eval_correct_count'])
        if trainer.state.best_metric != count or trainer.state.best_global_step != epoch*36:
            raise RuntimeError('Trainer selected a different best checkpoint')
        export_start = time.perf_counter()
        export_best(trainer.state.best_model_checkpoint, output/'best', identity, epoch, count)
        metrics.update(export_seconds=time.perf_counter()-export_start,
                       val_accuracy=count/2020, val_correct_count=count,
                       val_sample_count=2020, best_epoch=epoch)
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
                # A failed CUDA context must not prevent writing the original failure history.
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

        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value

        write_json(output/'history.json', clean(trainer.state.log_history if trainer else []))
        write_json(output/'updates.json', clean(audit.events))
        write_json(output/'metrics.json', metrics)
        record = status_record(status, reason)
        write_json(output/'training_status.json', record)
        # Future evaluation may change this progress file; sealed training files stay immutable.
        (output/'status.json').write_text(json.dumps(record, indent=2)+'\n')
    if error is not None:
        raise error
    seal_training(output, plan, run)
    return status


def main():
    from transformers import AutoImageProcessor, set_seed
    import torch
    from configs import MODEL_ID, load_model
    from data import read_manifest, official_train, ManifestDataset, collate_fn
    from run_queue import environment, validate_admission

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--executor', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; every retry requires a new attempt directory')
    plan = read_final_plan(args.plan, args.protocol, args.freeze)
    candidates = [r for r in plan['runs'] if r['run_id'] == args.run_id and r['executor'] == args.executor]
    if len(candidates) != 1:
        parser.error('Run is not assigned to this executor')
    run, common = candidates[0], plan['common']
    env = environment()
    # The unchanged Trainer, math profile and libraries retain the original GPU acceptance.
    validate_admission(args.admission, plan['lr_plan'], args.executor, env)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    set_seed(run['seed'])
    manifest = read_manifest(args.protocol/'data_manifest.json')
    source = official_train(common['dataset_revision'])
    if source.features['label'].names != manifest['label_names']:
        raise ValueError('Source label order differs from the common manifest')
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=common['model_revision'], use_fast=False)
    train = ManifestDataset(source, manifest, 'train', processor, run['seed'])
    val = ManifestDataset(source, manifest, 'val', processor, run['seed'])
    model = load_model(run['config'], run['seed'], common['model_revision'], manifest['label_names'])
    settings = training_arguments(plan, run, args.output)
    status = execute_final_training(model, train, val, settings, args.output, plan, run,
                                    env, collate_fn, processor)
    if status == 'numerical_invalid':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
