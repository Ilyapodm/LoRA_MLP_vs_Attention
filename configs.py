"""Four experiment configurations. No imports of torch/PEFT until model construction."""
from dataclasses import dataclass
import hashlib
import re

MODEL_ID = "google/vit-base-patch16-224-in21k"
NUM_LABELS = 101
LR_GRID = (1e-4, 1e-3, 1e-2)
SEARCH_SEED = 142
FINAL_SEEDS = (42, 43, 44)
EPOCHS = 5


@dataclass(frozen=True)
class Experiment:
    name: str
    r_attn: int
    r_mlp: int

    @property
    def lora_parameters(self):
        return 12 * (6144 * self.r_attn + 7680 * self.r_mlp)

    @property
    def trainable_parameters(self):
        return self.lora_parameters + 768 * NUM_LABELS + NUM_LABELS


CONFIGS = {name: Experiment(name, a, m) for name, a, m in (
    ("LP", 0, 0), ("A-iso", 20, 0), ("M-iso", 0, 16), ("B-iso", 10, 8)
)}


def targets(config):
    """Full names, ranks and (input, output) dimensions in Transformers 4.57.1."""
    result = {}
    for i in range(12):
        prefix = f"vit.encoder.layer.{i}."
        if config.r_attn:
            for suffix in ("attention.attention.query", "attention.attention.key",
                           "attention.attention.value", "attention.output.dense"):
                result[prefix + suffix] = (config.r_attn, 768, 768)
        if config.r_mlp:
            result[prefix + "intermediate.dense"] = (config.r_mlp, 768, 3072)
            result[prefix + "output.dense"] = (config.r_mlp, 3072, 768)
    return result


def lora_config(config):
    from peft import LoraConfig
    selected = targets(config)
    if not selected:
        raise ValueError("LP has no adapters")
    # PEFT pattern keys are regex: escape dots, and anchor the entire module name.
    patterns = {"^" + re.escape(name): spec[0] for name, spec in selected.items()}
    default_rank = config.r_attn or config.r_mlp
    return LoraConfig(
        r=default_rank, lora_alpha=default_rank,
        target_modules=list(selected), rank_pattern=patterns, alpha_pattern=patterns,
        lora_dropout=0.1, bias="none", modules_to_save=["classifier"],
        init_lora_weights=True, use_rslora=False, use_dora=False,
    )


def active_head(model):
    head = model.classifier
    return head.modules_to_save["default"] if hasattr(head, "modules_to_save") else head


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def head_hash(model):
    head = active_head(model)
    return hashlib.sha256((tensor_hash(head.weight) + tensor_hash(head.bias)).encode()).hexdigest()


def configure_model(base, config, seed):
    """Accept a fresh unwrapped ViT; isolate head/adapter RNG from the caller."""
    import torch
    from peft import get_peft_model
    c = base.config
    if (c.hidden_size, c.intermediate_size, c.num_hidden_layers,
        c.num_attention_heads, c.image_size, c.patch_size, c.num_labels) != (768, 3072, 12, 12, 224, 16, 101):
        raise ValueError("Unexpected ViT architecture or number of labels")
    if base.classifier.in_features != 768 or base.classifier.out_features != NUM_LABELS:
        raise ValueError("Expected a 768 -> 101 classifier")
    base.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        torch.nn.init.trunc_normal_(base.classifier.weight, mean=0,
                                   std=c.initializer_range, a=-2, b=2)
        torch.nn.init.zeros_(base.classifier.bias)
    initial_head = head_hash(base)
    modules = dict(base.named_modules())
    for name, (_, n_in, n_out) in targets(config).items():
        layer = modules.get(name)
        if not isinstance(layer, torch.nn.Linear) or (layer.in_features, layer.out_features) != (n_in, n_out):
            raise ValueError(f"Unexpected target module: {name}")
    if config.name == "LP":
        base.classifier.requires_grad_(True)
        model = base
    else:
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed + 10000)
            model = get_peft_model(base, lora_config(config))
    if head_hash(model) != initial_head:
        raise AssertionError("PEFT changed the initial classifier")
    audit_model(model, config)
    return model


def audit_model(model, config):
    """Check every adapter, alpha, scaling and exact trainable whitelist."""
    from peft.tuners.lora import LoraLayer
    selected = targets(config)
    allowed, seen = set(), set()
    head = active_head(model)
    allowed.update(id(p) for p in head.parameters())
    for name, layer in model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        canonical = name.removeprefix("base_model.model.")
        if canonical not in selected:
            raise AssertionError(f"Unexpected adapter: {name}")
        rank, n_in, n_out = selected[canonical]
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        if tuple(a.shape) != (rank, n_in) or tuple(b.shape) != (n_out, rank):
            raise AssertionError(f"Wrong adapter shapes: {name}")
        if layer.r["default"] != rank or layer.lora_alpha["default"] != rank or layer.scaling["default"] != 1:
            raise AssertionError(f"Wrong rank/alpha/scaling: {name}")
        allowed.update((id(a), id(b)))
        seen.add(canonical)
    if seen != set(selected):
        raise AssertionError(f"Missing adapters: {set(selected) - seen}")
    actual_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if actual_ids != allowed:
        raise AssertionError("Trainable whitelist mismatch")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n != config.trainable_parameters:
        raise AssertionError(f"{config.name}: {n} != {config.trainable_parameters}")
    return {"config": config.name, "trainable": n,
            "lora_parameters": config.lora_parameters, "head_parameters": 77669,
            "adapter_count": len(seen), "head_hash": head_hash(model)}


def load_model(name, seed, revision, label_names=None):
    """Only immutable Hub commits are accepted for pretrained runs."""
    from transformers import ViTForImageClassification
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Pass a full 40-character model commit SHA, not main")
    if label_names is None:
        label_names = [str(i) for i in range(NUM_LABELS)]  # Synthetic checks only.
    if len(label_names) != NUM_LABELS or len(set(label_names)) != NUM_LABELS:
        raise ValueError("Expected 101 unique label names")
    labels = dict(enumerate(label_names))
    base = ViTForImageClassification.from_pretrained(
        MODEL_ID, revision=revision, num_labels=NUM_LABELS,
        id2label=labels, label2id={v: k for k, v in labels.items()},
        attn_implementation="sdpa", torch_dtype="float32",
    )
    model = configure_model(base, CONFIGS[name], seed)
    if name != 'LP':
        model.peft_config['default'].revision = revision
    return model
