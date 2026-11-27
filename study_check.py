"""V2 GPU acceptance: all six parameter/gradient checks and real pretrained B-16."""
import argparse
import gc
from pathlib import Path
import time

from checkpoints import write_json
from study_configs import CONFIGS, configure_model, audit_model, active_head, load_model, MODEL_ID
from study_protocol import read_plan
from test_params import frozen_hash


def parameter_check(name, device, image_size=224):
    import torch
    from transformers import ViTConfig, ViTForImageClassification
    from peft.tuners.lora import LoraLayer
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(12345)
        base = ViTForImageClassification(ViTConfig(num_labels=101, attn_implementation='sdpa'))
    model = configure_model(base, CONFIGS[name], 42)
    report = audit_model(model, CONFIGS[name])
    before, head_before = frozen_hash(model), active_head(model).weight.detach().clone()
    for layer in model.modules():
        if isinstance(layer, LoraLayer) and torch.count_nonzero(layer.lora_B['default'].weight).item():
            raise AssertionError('Initial LoRA B is not zero')
    model.to(device).train()
    torch.manual_seed(20042)
    x = torch.randn(2, 3, image_size, image_size, device=device)
    y = torch.arange(2, device=device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=0)
    cuda = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)
    applied = attempts = skips = consecutive = 0
    a_gradient = b_gradient = head_gradient = False
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    while applied < 3:
        attempts += 1
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.float16, enabled=cuda):
            outputs = model(pixel_values=x, labels=y, interpolate_pos_encoding=image_size != 224)
            loss = outputs.loss
        if not torch.isfinite(loss).item() or not torch.isfinite(outputs.logits).all().item():
            raise AssertionError('Nonfinite synthetic forward')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite = all(torch.isfinite(p.grad).all().item()
                     for p in model.parameters() if p.requires_grad and p.grad is not None)
        if not finite:
            if not cuda:
                raise AssertionError('Nonfinite CPU gradients')
            scaler.step(optimizer)
            scaler.update()
            skips += 1
            consecutive += 1
            if consecutive >= 10:
                raise RuntimeError('10 consecutive AMP skips during v2 parameter acceptance')
            continue
        for key, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and torch.count_nonzero(p.grad).item():
                a_gradient |= 'lora_A.' in key
                b_gradient |= 'lora_B.' in key
                head_gradient |= 'classifier.' in key
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1)
        scaler.step(optimizer)
        scaler.update()
        applied += 1
        consecutive = 0
        if not all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
            raise AssertionError('Nonfinite updated weights')
    if cuda:
        torch.cuda.synchronize()
    report.update(stage='technical', scientific_result=False, seconds=time.perf_counter()-start,
                  applied_steps=applied, attempted_steps=attempts, skipped_steps=skips,
                  image_size=image_size, device=device,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None)
    model.cpu()
    if frozen_hash(model) != before:
        raise AssertionError('Frozen backbone changed')
    if not head_gradient or torch.equal(active_head(model).weight.detach(), head_before):
        raise AssertionError('Classifier failed to learn')
    if name != 'LP' and not (a_gradient and b_gradient):
        raise AssertionError('Missing LoRA A/B gradients')
    report['checks'] = ['exact_shapes_ranks_scaling_whitelist', 'same_initial_head', 'B_zero',
                        'head_updated', 'frozen_unchanged', 'finite_updates']
    if name != 'LP':
        report['checks'].append('A_and_B_gradients')
    del model, base, optimizer, scaler, x, y, outputs, loss, head_before
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return report


def technical_check(plan_path, protocol, output):
    import torch
    from transformers import AutoImageProcessor
    from data import read_manifest, official_train, ManifestDataset, collate_fn
    from train import ImageTrainer, UpdateAudit, PrefixDataset, classification_metrics, argmax_logits
    from study_train import _arguments, configure_runtime, clean_json
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('V2 real-data preflight requires exactly one visible CUDA GPU')
    plan = read_plan(plan_path, protocol)
    if plan['stage'] != 'lr_search':
        raise ValueError('Use the v2 LR plan for technical acceptance')
    common, output = plan['common'], Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh technical directory')
    configure_runtime(42)
    manifest = read_manifest(Path(protocol)/'data_manifest.json')
    source = official_train(common['dataset_revision'])
    if source.features['label'].names != manifest['label_names']:
        raise ValueError('Technical source labels differ')
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=common['model_revision'], use_fast=False)
    train = PrefixDataset(ManifestDataset(source, manifest, 'train', processor, 42), 772)
    val = PrefixDataset(ManifestDataset(source, manifest, 'val', processor, 42), 32)
    model = load_model('B-16', 42, common['model_revision'], manifest['label_names'])
    output.mkdir(parents=True)
    write_json(output/'technical-config.json', {'stage': 'technical', 'scientific_result': False,
        'plan_hash': plan['plan_hash'], 'config': 'B-16', 'seed': 42,
        'train_images': 772, 'val_images': 32, 'epochs': 2, 'profile': common['profile']})
    settings = _arguments(common, {'config': 'B-16', 'lr': 1e-3, 'seed': 42}, output, 2, False)
    audit = UpdateAudit()
    trainer = ImageTrainer(model=model, args=settings, train_dataset=train, eval_dataset=val,
        data_collator=collate_fn, processing_class=processor, compute_metrics=classification_metrics,
        preprocess_logits_for_metrics=argmax_logits, callbacks=[audit])
    audit.trainer = trainer
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        trainer.train()
    finally:
        write_json(output/'updates.json', clean_json(audit.events))
        write_json(output/'history.json', clean_json(trainer.state.log_history))
    torch.cuda.synchronize()
    evaluations = [h for h in trainer.state.log_history if 'eval_correct_count' in h]
    if (trainer.state.global_step != 4 or trainer.state.epoch != 2 or len(evaluations) != 2 or
            any(h['eval_sample_count'] != 32 for h in evaluations) or
            not any(not e['skipped'] for e in audit.events)):
        raise RuntimeError('Incomplete B-16 acceptance of full and partial accumulation groups')
    write_json(output/'technical-result.json', {'stage': 'technical', 'scientific_result': False,
        'status': 'passed', 'config': 'B-16', 'global_step': trainer.state.global_step,
        'wall_train_job_seconds': time.perf_counter()-start,
        'validation_seconds': trainer.validation_seconds, 'checkpoint_seconds': trainer.checkpoint_seconds,
        'applied_steps': sum(not e['skipped'] for e in audit.events),
        'skipped_steps': sum(e['skipped'] for e in audit.events),
        'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30})


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--parameters', action='store_true')
    mode.add_argument('--technical', action='store_true')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--plan', type=Path)
    p.add_argument('--protocol', type=Path, default=Path('artifacts/protocol'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; preserve the previous acceptance')
    if args.technical:
        if not args.plan:
            p.error('--technical needs --plan')
        technical_check(args.plan, args.protocol, args.output)
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; no automatic CPU fallback')
    from study_train import configure_runtime
    configure_runtime(42)
    reports = []
    for name in CONFIGS:
        reports.append(parameter_check(name, args.device))
        print(name, 'passed', flush=True)
    if len({r['head_hash'] for r in reports}) != 1:
        raise AssertionError('Initial classifiers differ across methods')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {'schema_version': 2, 'stage': 'technical',
                            'scientific_result': False, 'runs': reports})


if __name__ == '__main__':
    main()
