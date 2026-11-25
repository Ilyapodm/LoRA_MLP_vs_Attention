"""Actual PEFT/ViT smoke check, synthetic inputs only. Never produces scientific metrics."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import time


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_one(name, args):
    import torch
    from peft.tuners.lora import LoraLayer
    from transformers import ViTConfig, ViTForImageClassification
    from configs import CONFIGS, active_head, audit_model, configure_model, load_model
    if args.pretrained:
        model = load_model(name, args.seed, args.revision)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12345)  # Identical random base for each configuration.
            base = ViTForImageClassification(ViTConfig(num_labels=101, attn_implementation="sdpa"))
        model = configure_model(base, CONFIGS[name], args.seed)
    report = audit_model(model, CONFIGS[name])
    before = frozen_hash(model)
    head_before = active_head(model).weight.detach().clone()
    # B=0 is checked in every target, not just one representative.
    for layer in model.modules():
        if isinstance(layer, LoraLayer):
            if torch.count_nonzero(layer.lora_B['default'].weight).item():
                raise AssertionError('Initial LoRA B must be zero')
    model.to(args.device).train()
    torch.manual_seed(args.seed + 20000)
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=args.device)
    y = torch.arange(args.batch_size, device=args.device) % 101
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=1e-3, weight_decay=0)
    amp = args.device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    if amp:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    a_gradient = b_gradient = head_gradient = False
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.accumulation):
            with torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext():
                outputs = model(pixel_values=x, labels=y,
                                interpolate_pos_encoding=args.image_size != 224)
                loss = outputs.loss
            if not torch.isfinite(loss).item() or not torch.isfinite(outputs.logits).all().item():
                raise AssertionError('Nonfinite loss/logits')
            losses.append(float(loss.detach()))
            scaler.scale(loss / args.accumulation).backward()
        scaler.unscale_(optimizer)
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if not torch.isfinite(p.grad).all().item():
                    raise AssertionError(f'Nonfinite synthetic smoke gradient: {n}')
                nonzero = torch.count_nonzero(p.grad).item() > 0
                a_gradient |= 'lora_A.' in n and nonzero
                b_gradient |= 'lora_B.' in n and nonzero
                head_gradient |= 'classifier.' in n and nonzero
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        scaler.step(optimizer)
        scaler.update()
    if amp:
        torch.cuda.synchronize()
    report.update(seconds=time.perf_counter()-start, losses=losses,
                  image_size=args.image_size, batch_size=args.batch_size,
                  accumulation=args.accumulation, steps=args.steps, device=args.device,
                  pretrained=args.pretrained, model_revision=args.revision if args.pretrained else None,
                  stage='technical', scientific_result=False)
    if amp:
        report.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    model.cpu()
    if frozen_hash(model) != before:
        raise AssertionError('Frozen weights changed')
    if torch.equal(active_head(model).weight.detach(), head_before) or not head_gradient:
        raise AssertionError('Head did not learn')
    if name != 'LP' and not (a_gradient and b_gradient):
        raise AssertionError('Missing LoRA A/B gradients after several steps')
    report['checks'] = ['target_shapes', 'ranks_alpha_scaling', 'trainable_whitelist',
                        'initial_head_hash', 'initial_B_zero', 'finite_forward_backward',
                        'head_updated', 'frozen_weights_unchanged']
    if name != 'LP':
        report['checks'].append('A_and_B_gradients')
    del model, optimizer, scaler, x, y, outputs, loss, head_before
    gc.collect()
    if amp:
        torch.cuda.empty_cache()
    return report


def main():
    from configs import CONFIGS
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', choices=tuple(CONFIGS)+('all',), default='all')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--image-size', type=int, choices=(32,224), default=32)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--revision')
    parser.add_argument('--output', type=Path, default=Path('artifacts/smoke.json'))
    args = parser.parse_args()
    if args.steps < 2 or args.batch_size < 1 or args.accumulation < 1:
        parser.error('Need >=2 steps and positive batch/accumulation')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; no silent CPU fallback')
    if args.pretrained and (not args.revision or args.image_size != 224):
        parser.error('Pretrained check requires a commit SHA and --image-size 224')
    if args.output.exists():
        parser.error('Output exists; choose a new path to preserve previous check')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    reports = []
    names = tuple(CONFIGS) if args.config == 'all' else (args.config,)
    for name in names:
        result = run_one(name, args)
        reports.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if len({r['head_hash'] for r in reports}) != 1:
        raise AssertionError('Heads differ across configurations of the same seed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'stage':'technical', 'runs':reports}, indent=2)+'\n')


if __name__ == '__main__':
    main()
