from collections import OrderedDict
from collections.abc import Mapping
from typing import Optional, Tuple

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema,
    epoch: int,
    global_step: int,
    best_val: float,
    cfg,
) -> None:
    """Luu toan bo trang thai training vao 1 file .pt."""
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val": best_val,
        "cfg": cfg.__dict__,
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    ema=None,
    map_location: str = "cpu",
) -> Tuple[int, int, float]:
    """Nap checkpoint day du (model + optimizer + scheduler + ema neu co).

    Returns:
        (epoch, global_step, best_val) da luu trong checkpoint (mac dinh (0, 0, inf) neu khong co).
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0), ckpt.get(
        "best_val", float("inf")
    )


def load_model_only(
    path: str,
    model: torch.nn.Module,
    map_location: str = "cpu",
) -> None:
    """Nap EMA weights va xac minh checkpoint khop day du voi model.

    Nguon weights van la ``checkpoint["ema"]`` nhu implementation cu. Ham se
    kiem tra kien truc (neu checkpoint co ``cfg``), key, tensor type va shape
    truoc khi thay doi model; sau strict load se doi chieu lai tung tensor.
    """
    prefix = "[load_model_only]"

    def report(message: str) -> None:
        print(f"{prefix} {message}", flush=True)

    def preview(values, limit: int = 5) -> str:
        values = list(values)
        result = ", ".join(str(value) for value in values[:limit])
        if len(values) > limit:
            result += f", ... (+{len(values) - limit})"
        return result or "-"

    def normalize_config(value):
        if isinstance(value, (list, tuple, torch.Size)):
            return tuple(normalize_config(item) for item in value)
        return value

    report(
        f"START | path={path!r} | map_location={map_location!r} | "
        f"model={type(model).__name__}"
    )
    if not isinstance(model, torch.nn.Module):
        report(f"FAIL | model phai la torch.nn.Module, nhan {type(model).__name__}.")
        raise TypeError("model phai la torch.nn.Module.")
    if str(map_location).startswith("cuda"):
        report(
            "WARNING | map_location CUDA nap ca optimizer/scheduler len GPU; "
            "nen dung 'cpu' de giam nguy co OOM."
        )

    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        report(f"FAIL | khong doc duoc checkpoint: {type(exc).__name__}: {exc}")
        raise
    if not isinstance(checkpoint, Mapping):
        report(f"FAIL | checkpoint co type {type(checkpoint).__name__}, can mapping.")
        raise TypeError("Checkpoint phai la mapping co chua key 'ema'.")

    components = (
        f"{key!r}<{type(value).__name__}>" for key, value in checkpoint.items()
    )
    report(f"CHECKPOINT | {preview(components, limit=12)}")
    metadata = [
        f"{key}={checkpoint[key]!r}"
        for key in ("epoch", "global_step", "best_val")
        if key in checkpoint
    ]
    if metadata:
        report(f"METADATA | {' | '.join(metadata)}")

    if "ema" not in checkpoint:
        report(f"FAIL | khong co 'ema'; keys={preview(checkpoint.keys(), 12)}")
        raise KeyError(
            "Checkpoint khong co key 'ema'; ham khong tu fallback sang 'model' "
            "de tranh nap nham nguon weights."
        )
    state_dict = checkpoint["ema"]
    if state_dict is None:
        report("FAIL | checkpoint['ema'] la None.")
        raise ValueError("checkpoint['ema'] la None.")
    if not isinstance(state_dict, Mapping):
        report(
            "FAIL | checkpoint['ema'] phai la state_dict, "
            f"nhan {type(state_dict).__name__}."
        )
        raise TypeError("checkpoint['ema'] phai la state_dict.")
    if not state_dict:
        report("FAIL | checkpoint['ema'] la state_dict rong.")
        raise ValueError("checkpoint['ema'] la state_dict rong.")

    wrapper_types = (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    load_target = model.module if isinstance(model, wrapper_types) else model

    # Shape khop van co the sai semantics, vi du strides khong nam trong state_dict.
    architecture_fields = (
        "nc",
        "reg_max",
        "backbone_w",
        "backbone_n",
        "neck_n",
        "strides",
    )
    architecture_checks = []
    architecture_mismatches = []
    cfg = checkpoint.get("cfg")
    model_architecture_fields = [
        field for field in architecture_fields if hasattr(load_target, field)
    ]
    saved_model_fields = []
    if isinstance(cfg, Mapping):
        objects = [("model", load_target)]
        if isinstance(getattr(load_target, "head", None), torch.nn.Module):
            objects.append(("model.head", load_target.head))
        for object_name, current_object in objects:
            for field in architecture_fields:
                if field not in cfg or not hasattr(current_object, field):
                    continue
                saved = normalize_config(cfg[field])
                current = normalize_config(getattr(current_object, field))
                label = f"{object_name}.{field}"
                architecture_checks.append(label)
                if object_name == "model":
                    saved_model_fields.append(field)
                if saved != current:
                    architecture_mismatches.append(
                        f"{label}: checkpoint={saved!r}, model={current!r}"
                    )

    if architecture_mismatches:
        report(
            f"ARCH FAIL | mismatch={len(architecture_mismatches)} | "
            f"{preview(architecture_mismatches)}"
        )
        raise RuntimeError(
            "Kien truc model khong khop checkpoint['cfg']; model chua bi thay doi."
        )
    if not model_architecture_fields or not saved_model_fields:
        architecture_status = "unknown"
    elif set(saved_model_fields) == set(model_architecture_fields):
        architecture_status = "cfg_matched"
    else:
        architecture_status = "partial"
    report(
        f"ARCH {'PASS' if architecture_status == 'cfg_matched' else 'WARNING'} | "
        f"cfg_fields={len(saved_model_fields)}/{len(model_architecture_fields)} | "
        f"attributes_checked={len(architecture_checks)} | status={architecture_status}"
    )

    target_state = load_target.state_dict()
    if not target_state:
        report("FAIL | model.state_dict() rong.")
        raise ValueError("model.state_dict() rong.")
    invalid_keys = [repr(key) for key in state_dict if not isinstance(key, str)]
    if invalid_keys:
        report(f"FAIL | state_dict co key khong phai str: {preview(invalid_keys)}")
        raise TypeError("Moi key trong checkpoint['ema'] phai la str.")

    source_keys = set(state_dict)
    target_keys = set(target_state)
    if source_keys and all(key.startswith("module.") for key in source_keys):
        stripped = {key[len("module."):] for key in source_keys}
        if len(stripped & target_keys) > len(source_keys & target_keys):
            source_metadata = getattr(state_dict, "_metadata", None)
            state_dict = OrderedDict(
                (key[len("module."):], value) for key, value in state_dict.items()
            )
            if source_metadata:
                state_dict._metadata = {
                    ("" if key == "module" else key[len("module."):]): value
                    for key, value in source_metadata.items()
                    if key == "module" or key.startswith("module.")
                }
            source_keys = set(state_dict)
            report("NORMALIZE | da bo prefix DataParallel/DDP 'module.'.")

    source_non_tensors = sorted(
        key for key, value in state_dict.items() if not isinstance(value, torch.Tensor)
    )
    target_non_tensors = sorted(
        key for key, value in target_state.items() if not isinstance(value, torch.Tensor)
    )
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    bad_shapes = []
    dtype_conversions = []
    compatible = set()
    for key in source_keys & target_keys:
        source = state_dict[key]
        target = target_state[key]
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            continue
        if source.shape != target.shape:
            bad_shapes.append(
                f"{key}: checkpoint{tuple(source.shape)} != model{tuple(target.shape)}"
            )
        else:
            compatible.add(key)
            if source.dtype != target.dtype:
                dtype_conversions.append(f"{key}: {source.dtype} -> {target.dtype}")
    bad_shapes.sort()
    dtype_conversions.sort()

    target_tensor_keys = {
        key for key, value in target_state.items() if isinstance(value, torch.Tensor)
    }
    target_elements = sum(target_state[key].numel() for key in target_tensor_keys)
    compatible_elements = sum(target_state[key].numel() for key in compatible)
    tensor_coverage = len(compatible) / max(len(target_tensor_keys), 1)
    element_coverage = compatible_elements / max(target_elements, 1)
    has_errors = bool(
        missing
        or unexpected
        or bad_shapes
        or source_non_tensors
        or target_non_tensors
    )
    report(
        f"PREFLIGHT {'FAIL' if has_errors else 'PASS'} | source=checkpoint['ema'] | "
        f"tensors={len(compatible):,}/{len(target_tensor_keys):,} "
        f"({tensor_coverage:.2%}) | elements={compatible_elements:,}/"
        f"{target_elements:,} ({element_coverage:.2%}) | missing={len(missing)} | "
        f"unexpected={len(unexpected)} | bad_shape={len(bad_shapes)} | "
        f"non_tensor={len(source_non_tensors) + len(target_non_tensors)} | "
        f"dtype_conversion={len(dtype_conversions)}"
    )

    component_names = sorted(
        {key.split(".", 1)[0] for key in source_keys | target_keys}
    )
    for component in component_names[:20]:
        target_component = {
            key for key in target_tensor_keys if key.split(".", 1)[0] == component
        }
        matched_component = target_component & compatible
        target_component_elements = sum(
            target_state[key].numel() for key in target_component
        )
        matched_component_elements = sum(
            target_state[key].numel() for key in matched_component
        )
        key_ratio = len(matched_component) / max(len(target_component), 1)
        element_ratio = matched_component_elements / max(target_component_elements, 1)
        report(
            f"COMPONENT | {component} | tensors={len(matched_component):,}/"
            f"{len(target_component):,} ({key_ratio:.2%}) | "
            f"elements={matched_component_elements:,}/{target_component_elements:,} "
            f"({element_ratio:.2%})"
        )

    details = (
        ("missing keys", missing),
        ("unexpected keys", unexpected),
        ("bad shapes", bad_shapes),
        ("non-tensor checkpoint values", source_non_tensors),
        ("non-tensor model values", target_non_tensors),
    )
    for label, values in details:
        if values:
            report(f"DETAIL | {label}: {preview(values)}")
    if dtype_conversions:
        report(f"WARNING | dtype conversions: {preview(dtype_conversions)}")
    if has_errors:
        raise RuntimeError(
            "checkpoint['ema'] khong khop 100% voi model; model chua bi thay "
            "doi vi loi duoc phat hien o buoc preflight."
        )

    try:
        load_target.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        report(f"LOAD FAIL | {type(exc).__name__}: {exc}")
        raise

    loaded_state = load_target.state_dict()
    verification_failures = []
    for key in sorted(target_tensor_keys):
        loaded = loaded_state[key].detach()
        expected = state_dict[key].detach().to(loaded.device, loaded.dtype)
        equal = torch.equal(loaded, expected)
        if not equal and (loaded.dtype.is_floating_point or loaded.dtype.is_complex):
            equal = torch.allclose(
                loaded, expected, rtol=0.0, atol=0.0, equal_nan=True
            )
        if not equal:
            verification_failures.append(key)
    if verification_failures:
        report(f"VERIFY FAIL | values: {preview(verification_failures)}")
        raise RuntimeError("Hau kiem phat hien weights da nap khong khop checkpoint.")

    value_status = "exact_after_target_dtype_cast" if dtype_conversions else "exact"
    report(
        f"VERIFY PASS | tensors={len(target_tensor_keys):,}/"
        f"{len(target_tensor_keys):,} | elements={target_elements:,}/"
        f"{target_elements:,} | values={value_status}"
    )
    report(
        f"PASS | source=checkpoint['ema'] | state_dict_coverage=100.00% | "
        f"cfg_architecture={architecture_status} | "
        f"dtype_conversions={len(dtype_conversions)}"
    )
