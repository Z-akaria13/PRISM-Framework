import torch
import numpy as np
from typing import Dict, List, Any, Optional

from prism.core.provenance import weights_digest
from prism.inspectors.base import Inspector


def _move_inputs_to_model_device(module, args, kwargs):
    """Forward pre-hook: connectors build inputs on the CPU; move them to wherever the model lives."""
    try:
        # Find what desk the model is sitting on
        device = next(module.parameters()).device
    except StopIteration:
        return args, kwargs

    # Helper function to find tensors and move them
    def move_to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(move_to_device(v) for v in obj)
        return obj

    return move_to_device(args), move_to_device(kwargs)


def _assemble_calls(name, calls, expected_rows):
    """{dataset name: arrays to join along rows} for one layer's (pass, call, array) records."""
    if any(p > 0 for p, _, _ in calls):  # whole-model passes were seen: group by position in the pass
        per_pass = {}
        for p, call, array in calls:
            per_pass.setdefault(p, []).append(array)
        counts = {len(arrays) for arrays in per_pass.values()}
        if len(counts) > 1:
            raise ValueError(f"'{name}' ran a different number of times in different forward passes "
                             f"({sorted(counts)}), so its calls cannot be lined up")
        k = counts.pop()
        ordered = [per_pass[p] for p in sorted(per_pass)]
        if k == 1:
            return {name: [arrays[0] for arrays in ordered]}
        return {f"{name}@{i}": [arrays[i] for arrays in ordered] for i in range(k)}

    arrays = [array for _, _, array in calls]  # submodules called directly: decide by row counts
    rows = [len(array) for array in arrays]
    if expected_rows is None or len(arrays) == 1 or sum(rows) == expected_rows:
        return {name: arrays}
    if all(r == expected_rows for r in rows):
        return {f"{name}@{i}": [array] for i, array in enumerate(arrays)}
    raise ValueError(f"'{name}' produced row counts {rows} for {expected_rows} items, which cannot be "
                     "split into separate calls")


class PyTorchInspector(Inspector):
    """
    Handles introspection of PyTorch models.
    
    Responsibilities:
    1. Walking the model graph to discover 'leaf' modules (layers).
    2. Attaching/Detaching forward hooks to capture intermediate activations.
    3. Managing the lifecycle of captured data (GPU -> CPU -> Numpy).
    """

    def __init__(self, model):
        super().__init__(model)
        self.hooks = []
        self._was_training = []  # modules switched to eval mode for a run, restored by teardown
        self._pass, self._calls_in_pass = 0, {}  # forward passes seen, and calls per layer in the current one

    def to_device(self, device: str = "auto") -> str:
        """
        Moves the model to `device` ("auto" = CUDA when this machine has it) and returns it.
        Called by the engine at the start of a run, so listing layers in the Setup tab
        never needs a GPU and cluster jobs use the compute node's hardware.
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        return device

    def prepare_for_inference(self) -> Optional[str]:
        """
        Puts every module in eval mode for the run (dropout off, batch norm on its stored
        statistics), so activations don't depend on randomness or on which rows share a batch.
        Remembers which modules were training so teardown() can put them back.
        Returns a warning if the connector left any module in training mode.
        """
        self._was_training = [module for module in self.model.modules() if module.training]
        if not self._was_training:
            return None
        self.model.eval()
        return (f"The model was in training mode ({len(self._was_training)} modules); PRISM switched it "
                "to eval mode for this run so dropout and batch norm behave as at inference. "
                "Call model.eval() in your connector's load_model() to avoid this warning.")

    def inference_context(self):
        """Wraps each run_inference call: no autograd graph, so a connector that forgets
        torch.no_grad() can't fill memory with gradients."""
        return torch.no_grad()


    def discover_topology(self) -> List[Dict[str, Any]]:
        """
        Scans the model for probe-able layers. Returns a flat list of metadata
        suitable for populating the UI TreeView.
        
        Note: We filter for 'Leaf' modules (layers with no children) to prevent
        double-counting parameters or probing container classes like Sequential.
        """
        layers = []
        
        for name, module in self.model.named_modules():
            if name == "": continue
            
            # Identify leaf nodes (e.g. Conv2d is a leaf, Sequential is not)
            is_leaf = len(list(module.children())) == 0
            
            if is_leaf:
                param_count = sum(p.numel() for p in module.parameters())
                layers.append({
                    "id": name,
                    "type": module.__class__.__name__,
                    "params": f"{param_count:,}"
                })
        
        return layers

    def attach_hooks(self, layer_names: List[str]):
        """
        Registers forward hooks on the specific layers requested.
        Automatically clears any existing hooks before attaching new ones.
        """
        self._remove_hooks()
        self.hooks.append(self.model.register_forward_pre_hook(self._start_pass))
        self.hooks.append(self.model.register_forward_pre_hook(_move_inputs_to_model_device, with_kwargs=True))

        # Create a set for O(1) lookup during graph walk
        target_layers = set(layer_names)
        
        for name, module in self.model.named_modules():
            if name in target_layers:
                hook_fn = self._create_hook_closure(name)
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)

    def model_fingerprint(self) -> str:
        """The weights and buffers (e.g. batch norm statistics), wherever the model sits and whatever
        its mode."""
        def entry(name, tensor):
            tensor = tensor.detach().cpu()
            data = (tensor.float() if tensor.dtype == torch.bfloat16 else tensor).contiguous().numpy().tobytes()
            return name, tensor.dtype, tuple(tensor.shape), data
        return weights_digest(entry(name, tensor) for name, tensor in self.model.state_dict().items()
                              if isinstance(tensor, torch.Tensor))

    def output_shapes(self, layer_names: List[str], run) -> Dict[str, tuple]:
        """Like the base class, but the hooks keep only each output's shape, so showing the shapes
        of a large 3-D model's layers doesn't hold a copy of every activation."""
        shapes, handles = {}, []
        targets = set(layer_names)

        def recorder(name):
            def record(module, args, output):
                data = output[0] if isinstance(output, tuple) else output
                if isinstance(data, torch.Tensor):
                    shapes.setdefault(name, []).append(tuple(data.shape[1:]))
            return record

        handles.append(self.model.register_forward_pre_hook(_move_inputs_to_model_device, with_kwargs=True))
        for name, module in self.model.named_modules():
            if name in targets:
                handles.append(module.register_forward_hook(recorder(name)))
        try:
            run()
        finally:
            for handle in handles:
                handle.remove()
        return {(f"{name}@{i}" if len(calls) > 1 else name): shape
                for name, calls in shapes.items() for i, shape in enumerate(calls)}

    def retrieve_features(self, expected_rows: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        The activations captured since the buffer was cleared, one array per layer, rows in
        input order. Should be called after inference completes for a batch.

        A module that runs k > 1 times within one forward pass of the model (e.g. the ReLU a
        ResNet block applies twice) gives one array per call, named "name@0" ... "name@<k-1>".
        Separate passes (a connector looping over items, or running a batch in chunks) are
        joined along the rows. If the connector calls submodules directly, so whole-model
        passes are never seen, calls are told apart by `expected_rows` instead.
        """
        results = {}
        for name, calls in self._buffer.items():
            for call_name, arrays in _assemble_calls(name, calls, expected_rows).items():
                results[call_name] = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
        return results

    def clear_buffer(self):
        """
        Clears captured data without removing hooks. 
        """
        self._buffer = {}
        self._pass, self._calls_in_pass = 0, {}

    def _start_pass(self, module, args):
        """Forward pre-hook on the whole model: a new pass begins, so call counts restart."""
        self._pass += 1
        self._calls_in_pass = {}

    def teardown(self):
        """
        Leaves the user's model as it was: removes all hooks and restores the training
        mode of any module prepare_for_inference() switched to eval.
        """
        self._remove_hooks()
        for module in self._was_training:
            module.training = True  # per module, so a mix of train/eval modes comes back exactly
        self._was_training = []

    def _remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self._buffer = {}
        self._pass, self._calls_in_pass = 0, {}

    def _create_hook_closure(self, layer_name: str):
        """
        Generates the actual hook function. 
        Handles Tuple outputs (common in Transformers) and GPU->CPU transfer.
        """
        def hook(module, input, output):
            # layers that return (output, hidden_state) tuples
            if isinstance(output, tuple):
                data = output[0]
            else:
                data = output

            #can only probe Tensors
            if not isinstance(data, torch.Tensor):
                return

            #  Detach from graph, move to CPU, convert to Numpy. must happen immediately to prevent VRAM explosion. ask me how I know....
            cpu_data = data.detach().cpu().numpy()

            # Tag each call with its forward pass and its position within that pass
            call = self._calls_in_pass.get(layer_name, 0)
            self._calls_in_pass[layer_name] = call + 1
            self._buffer.setdefault(layer_name, []).append((self._pass, call, cpu_data))
            
        return hook