import torch
import torch.nn as nn

from ..vram.disk_map import DiskMap
from ..vram.initialization import skip_model_initialization
from ..vram.layers import enable_vram_management
from .file import load_state_dict


@torch.no_grad()
def materialize_module_meta_only(module: nn.Module, device, dtype):
    # parameters
    for name, p in list(module._parameters.items()):
        if p is not None and getattr(p, "is_meta", False):
            new_p = torch.empty(p.shape, device=device, dtype=dtype)
            module._parameters[name] = nn.Parameter(new_p, requires_grad=p.requires_grad)

    # buffers
    for name, b in list(module._buffers.items()):
        if b is not None and getattr(b, "is_meta", False):
            module._buffers[name] = torch.empty(b.shape, device=device, dtype=dtype)

    # recurse
    for child in module.children():
        materialize_module_meta_only(child, device, dtype)

def load_model(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, use_disk_map=False, module_map=None, vram_config=None, vram_limit=None):
    config = {} if config is None else config
    # Why do we use `skip_model_initialization`?
    # It skips the random initialization of model parameters,
    # thereby speeding up model loading and avoiding excessive memory usage.
    with skip_model_initialization():
        model = model_class(**config)
    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None:
        devices = [vram_config["offload_device"], vram_config["onload_device"], vram_config["preparing_device"], vram_config["computation_device"]]
        device = [d for d in devices if d != "disk"][0]
        dtypes = [vram_config["offload_dtype"], vram_config["onload_dtype"], vram_config["preparing_dtype"], vram_config["computation_dtype"]]
        dtype = [d for d in dtypes if d != "disk"][0]
        if vram_config["offload_device"] != "disk":
            state_dict = DiskMap(path, device, torch_dtype=dtype)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            else:
                state_dict = {i: state_dict[i] for i in state_dict}
            model.load_state_dict(state_dict, assign=True)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit)
        else:
            disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit)
    else:
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        if use_disk_map:
            state_dict = DiskMap(path, device, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict(path, torch_dtype, device)
        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        else:
            state_dict = {i: state_dict[i] for i in state_dict}
        missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
        # Why do we call `to()`?
        # Because some models override the behavior of `to()`,
        # especially those from libraries like Transformers.
        if hasattr(model, "gate_mlps"):
            # 只有当 gate_mlps 的某些权重缺失时，才 materialize
            need_gate = any(k.startswith("gate_mlps.") for k in missing)
            if need_gate:
                materialize_module_meta_only(model.gate_mlps, device=device, dtype=torch_dtype)

                # 只初始化 gate_mlps（此时它们是新分配的 empty tensor）
                for g in model.gate_mlps:
                    if hasattr(g, "post_init"):
                        g.post_init()
        if hasattr(model, "routers"):
            # 只有当 gate_mlps 的某些权重缺失时，才 materialize
            need_gate = any(k.startswith("routers.") for k in missing)
            if need_gate:
                materialize_module_meta_only(model.routers, device=device, dtype=torch_dtype)

                # 只初始化 gate_mlps（此时它们是新分配的 empty tensor）
                for g in model.routers:
                    if hasattr(g, "post_init"):
                        g.post_init()
        if hasattr(model, "single_gate_mlps"):
            # 只有当 gate_mlps 的某些权重缺失时，才 materialize
            need_gate = any(k.startswith("single_gate_mlps.") for k in missing)
            if need_gate:
                materialize_module_meta_only(model.single_gate_mlps, device=device, dtype=torch_dtype)

                # 只初始化 gate_mlps（此时它们是新分配的 empty tensor）
                for g in model.single_gate_mlps:
                    if hasattr(g, "post_init"):
                        g.post_init()

                        
        model = model.to(dtype=torch_dtype, device=device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def load_model_with_disk_offload(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, module_map=None):
    if isinstance(path, str):
        path = [path]
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_class(**config)
    if hasattr(model, "eval"):
        model = model.eval()
    disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": device,
        "computation_dtype": torch_dtype,
        "computation_device": device,
    }
    enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=80)
    return model
