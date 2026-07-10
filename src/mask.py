import torch
import torch.nn as nn

from src.layerwrapper import WrappedGPT


def tensor_mask_of_largest_elements(tensor: torch.tensor, k: int) -> torch.tensor:
    # Flatten the tensor and get the indices of the top k elements
    flat_tensor = tensor.view(-1)
    _, topk_indices = torch.topk(flat_tensor, k)

    # Create a boolean mask of the same shape as tensor
    mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
    mask[topk_indices] = True
    mask = mask.view(tensor.shape)

    return mask


def select_rowwise_bottom_then_global_fill(metric: torch.Tensor, sparse_rate: float) -> torch.Tensor:
    rows, cols = metric.shape
    target_num = min(int(sparse_rate * rows * cols), metric.numel())
    if target_num <= 0:
        return torch.empty(0, dtype=torch.long, device=metric.device)

    row_train_num = min(int(sparse_rate * cols), cols)
    if row_train_num > 0:
        row_indices = torch.topk(metric, k=row_train_num, dim=1, largest=False).indices
        row_offsets = torch.arange(rows, device=metric.device, dtype=torch.long).reshape(-1, 1) * cols
        selected_indices = (row_offsets + row_indices.to(torch.long)).reshape(-1)
    else:
        selected_indices = torch.empty(0, dtype=torch.long, device=metric.device)

    fill_num = target_num - selected_indices.numel()
    if fill_num <= 0:
        return selected_indices[:target_num]

    flat_metric = metric.reshape(-1)
    remaining_metric = flat_metric.clone()
    if selected_indices.numel() > 0:
        remaining_metric[selected_indices] = torch.inf
    fill_indices = torch.topk(remaining_metric, k=fill_num, largest=False).indices
    return torch.cat([selected_indices, fill_indices])


def get_all_blocks(model):
    if "opt" not in model.name_or_path:
        return model.model.layers
    else:
        return model.model.decoder.layers


def find_layers(block, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        block (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """

    if type(block) in layers:
        return {name: block}
    res = {}

    for name1, child in block.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def prepare_super_mask(
        model,
        tokenizer,
        dev,
        sparse_rate=None,
        nsamples=128,
        seed=228,
        calibration_data="c4",
        outliers_ratio=None,
        collect_stats=False,
        max_layers=None,
        metric_order="top",
        hybrid_top_ratio=None,
):
    from src.datasets_loader import get_loaders

    if metric_order not in {"top", "bottom", "bottom-structured", "hybrid"}:
        raise ValueError("metric_order must be 'top', 'bottom', 'bottom-structured', or 'hybrid'.")
    if metric_order == "hybrid":
        if hybrid_top_ratio is None:
            raise ValueError("hybrid_top_ratio is required when metric_order='hybrid'.")
        if not 0.0 <= hybrid_top_ratio <= 1.0:
            raise ValueError("hybrid_top_ratio must be in [0, 1].")
    if sparse_rate is None:
        sparse_rate = outliers_ratio
    if sparse_rate is None:
        raise ValueError("prepare_super_mask requires sparse_rate or outliers_ratio.")
    dataloader, _ = get_loaders(calibration_data, nsamples, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer)
    stats = {
        "calibration_data": calibration_data,
        "requested_nsamples": nsamples,
        "actual_nsamples": 0,
        "seed": seed,
        "sparse_rate": float(sparse_rate),
        "metric_order": metric_order,
        "hybrid_top_ratio": None if hybrid_top_ratio is None else float(hybrid_top_ratio),
        "layers": [],
    } if collect_stats else None

    use_cache = model.config.use_cache
    model.config.use_cache = False

    blocks = get_all_blocks(model)

    dev = model.device

    #if "model.embed_tokens" in model.hf_device_map:
    #    dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, min(2048, model.seqlen), model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            if 'position_embeddings' in kwargs:
                cache['position_embeddings'] = kwargs['position_embeddings']
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    blocks[0] = Catcher(blocks[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    if stats is not None:
        stats["actual_nsamples"] = int(cache["i"])
    blocks[0] = blocks[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    block_args = {}
    if attention_mask is not None:
        block_args["attention_mask"] = attention_mask  # Add attention mask if defined
    if position_ids is not None:
        block_args["position_ids"] = position_ids  # Add position IDs if defined
    if position_embeddings is not None:
        block_args["position_embeddings"] = position_embeddings  # Add position embeddings if defined

    num_blocks = len(blocks) if max_layers is None else min(max_layers, len(blocks))
    for i in range(num_blocks):
        block = blocks[i]
        #if f"model.layers.{i}" in model.hf_device_map:
        #    dev = model.hf_device_map[f"model.layers.{i}"]
        #    print(f"layer {i} device {dev}")
        #    inps, outs = inps.to(dev), outs.to(dev)

        #    if attention_mask is not None:
        #        attention_mask = attention_mask.to(dev)
        #    if position_ids is not None:
        #        position_ids = position_ids.to(dev)

        subset = find_layers(block)

        wrappers = {}
        for name in subset:
            wrappers[name] = WrappedGPT(subset[name])

        def add_batch(_name):
            def tmp(_, inp, out):
                wrappers[_name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrappers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            outs[j] = block(inps[j].to(dev).unsqueeze(0), **block_args)[0]

        for h in handles:
            h.remove()

        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrappers[name].scaler_row.reshape((1, -1)))

            flat_tensor = W_metric.view(-1)

            in_features, out_features = subset[name].weight.shape

            #train_num = (out_features + in_features) * r
            train_num = min(int(sparse_rate * subset[name].weight.numel()) + 1, subset[name].weight.numel())

            if metric_order == "hybrid":
                top_train_num = min(train_num, int(round(train_num * hybrid_top_ratio)))
                bottom_train_num = train_num - top_train_num
                row_train_num = 0
                global_fill_train_num = 0
                selected_parts = []
                if top_train_num > 0:
                    selected_parts.append(torch.topk(flat_tensor, k=top_train_num, largest=True).indices)
                if bottom_train_num > 0:
                    selected_parts.append(torch.topk(flat_tensor, k=bottom_train_num, largest=False).indices)
                selected_indices = torch.cat(selected_parts)
            elif metric_order == "bottom-structured":
                selected_indices = select_rowwise_bottom_then_global_fill(W_metric, sparse_rate)
                train_num = int(selected_indices.numel())
                top_train_num = 0
                bottom_train_num = train_num
                row_train_num = int(sparse_rate * W_metric.shape[1])
                global_fill_train_num = train_num - row_train_num * W_metric.shape[0]
            else:
                top_train_num = train_num if metric_order == "top" else 0
                bottom_train_num = train_num if metric_order == "bottom" else 0
                row_train_num = 0
                global_fill_train_num = 0
                selected_indices = torch.topk(
                    flat_tensor,
                    k=train_num,
                    largest=(metric_order == "top"),
                ).indices
            selected_indices_cpu = selected_indices.cpu()
            subset[name].weight.wanda_selected_indices = selected_indices_cpu
            if metric_order == "top":
                subset[name].weight.wanda_topk_indices = selected_indices_cpu
            elif metric_order == "bottom":
                subset[name].weight.wanda_bottomk_indices = selected_indices_cpu
            elif metric_order == "bottom-structured":
                subset[name].weight.wanda_bottom_structured_indices = selected_indices_cpu
            else:
                subset[name].weight.wanda_hybrid_indices = selected_indices_cpu

            if stats is not None:
                scaler = wrappers[name].scaler_row.detach()
                selected_metric = flat_tensor[selected_indices].detach()
                stats["layers"].append(
                    {
                        "layer": int(i),
                        "name": name,
                        "metric_order": metric_order,
                        "hybrid_top_ratio": None if hybrid_top_ratio is None else float(hybrid_top_ratio),
                        "weight_shape": [int(dim) for dim in subset[name].weight.shape],
                        "train_num": int(train_num),
                        "top_train_num": int(top_train_num),
                        "bottom_train_num": int(bottom_train_num),
                        "row_train_num": int(row_train_num),
                        "global_fill_train_num": int(global_fill_train_num),
                        "numel": int(subset[name].weight.numel()),
                        "scaler_finite": bool(torch.isfinite(scaler).all().item()),
                        "scaler_nan_count": int(torch.isnan(scaler).sum().item()),
                        "scaler_inf_count": int(torch.isinf(scaler).sum().item()),
                        "scaler_min": float(scaler.min().item()),
                        "scaler_mean": float(scaler.mean().item()),
                        "scaler_max": float(scaler.max().item()),
                        "scaler_zero_frac": float((scaler == 0).float().mean().item()),
                        "metric_finite": bool(torch.isfinite(flat_tensor).all().item()),
                        "metric_nan_count": int(torch.isnan(flat_tensor).sum().item()),
                        "metric_inf_count": int(torch.isinf(flat_tensor).sum().item()),
                        "metric_min": float(flat_tensor.min().item()),
                        "metric_mean": float(flat_tensor.mean().item()),
                        "metric_max": float(flat_tensor.max().item()),
                        "selected_metric_min": float(selected_metric.min().item()),
                        "selected_metric_mean": float(selected_metric.mean().item()),
                        "selected_metric_max": float(selected_metric.max().item()),
                        "selected_unique_count": int(torch.unique(selected_indices).numel()),
                    }
                )

            print(i, name)

        blocks[i] = block
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    if stats is not None:
        return stats
