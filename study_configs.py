"""Protocol v2: equal-parameter and rank-16 studies share one canonical M-iso."""
from configs import (Experiment, MODEL_ID, NUM_LABELS, configure_model, audit_model,
                     active_head, head_hash, targets)

PROTOCOL_VERSION = 'food101-local-v2'
EPOCHS = 10
LR_GRID = (1e-4, 1e-3, 1e-2, 3e-2)
SEARCH_SEED = 142
FINAL_SEEDS = (42, 43, 44)
CONFIGS = {name: Experiment(name, a, m) for name, a, m in (
    ('LP', 0, 0), ('A-iso', 20, 0), ('M-iso', 0, 16),
    ('B-iso', 10, 8), ('A-16', 16, 0), ('B-16', 16, 16))}
ALIASES = {'M-16': 'M-iso'}
SERIES = {'iso': ['A-iso', 'M-iso', 'B-iso'],
          'rank16': ['A-16', 'M-iso', 'B-16']}
RECIPE = {
    'optimizer': 'adamw_torch', 'betas': [0.9, 0.999], 'epsilon': 1e-8,
    'weight_decay': 0.0, 'scheduler': 'linear', 'warmup_steps': 0,
    'max_grad_norm': 1.0, 'label_smoothing': 0.0,
    'lora_dropout': 0.1, 'lora_bias': 'none', 'lora_scaling': 'alpha=r',
    'lora_initialization': 'peft-default-B-zero', 'head_lr': 'same-as-adapters',
    'head_initialization': 'trunc-normal-0-std-config-bounds-minus2-plus2-bias-zero',
    'best_metric': 'correct_count', 'epoch_tie_break': 'earlier',
    'lr_tie_break': 'lower', 'early_stopping': False, 'evaluation_every_epochs': 1,
}


def canonical_config(name):
    name = ALIASES.get(name, name)
    if name not in CONFIGS:
        raise ValueError('Unknown v2 configuration: '+str(name))
    return name


def configuration_specs():
    return {name: {'r_attn': c.r_attn, 'r_mlp': c.r_mlp,
                   'lora_parameters': c.lora_parameters,
                   'trainable_parameters': c.trainable_parameters,
                   'adapter_count': len(targets(c))} for name, c in CONFIGS.items()}


def statistics_spec():
    from itertools import combinations
    return {
        'primary': 'test-top1-mean-and-sample-std-ddof1-across-seeds',
        'mcnemar': 'two-sided-exact', 'multiplicity': 'Holm-over-all-18-tests',
        'hypotheses': [{'series': series, 'left': left, 'right': right, 'seed': seed}
                       for series, methods in SERIES.items()
                       for left, right in combinations(methods, 2) for seed in FINAL_SEEDS],
        'lp_comparisons': 'descriptive-baseline-no-additional-p-values',
        'bootstrap': {'replicates': 5000, 'seed': 2026, 'stratified_by': 'class',
                      'unit': 'image', 'paired_indices': True,
                      'average_difference_over_seeds_before_resampling': True,
                      'interval': 'percentile-95-descriptive-no-simultaneous-coverage',
                      'interpretation': 'conditional-on-the-three-trained-seeds'},
    }


def load_model(name, seed, revision, label_names):
    """Fresh pretrained backbone/head/adapters; v1 model assembly is reused unchanged."""
    import re
    from transformers import ViTForImageClassification
    name = canonical_config(name)
    if not re.fullmatch('[0-9a-f]{40}', revision or ''):
        raise ValueError('Expected an immutable model commit SHA')
    if len(label_names) != NUM_LABELS or len(set(label_names)) != NUM_LABELS:
        raise ValueError('Expected 101 unique class names')
    labels = dict(enumerate(label_names))
    base = ViTForImageClassification.from_pretrained(
        MODEL_ID, revision=revision, num_labels=NUM_LABELS,
        id2label=labels, label2id={v: k for k, v in labels.items()},
        attn_implementation='sdpa', torch_dtype='float32')
    model = configure_model(base, CONFIGS[name], seed)
    if name != 'LP':
        model.peft_config['default'].revision = revision
    return model
