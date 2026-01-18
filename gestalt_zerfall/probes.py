import torch
import torch.nn as nn

class HookManager:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def register_hooks(self, layer_names=None):
        """
        Registers forward hooks on the specified layers.
        If layer_names is None, attempts to auto-detect Transformer layers.
        """
        # Clear previous hooks/activations
        self.clear_hooks()
        self.activations = {}

        # Auto-detection for GPT-2/BERT style models
        # This is a heuristic; might need adjustment for specific architectures
        modules_to_hook = []
        
        # Auto-detection for GPT-2/BERT/Mamba style models
        modules_to_hook = []
        
        # GPT-2
        if hasattr(self.model, 'h'):
            for i, layer in enumerate(self.model.h):
                modules_to_hook.append((f"layer_{i}", layer))
        # BERT
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
             for i, layer in enumerate(self.model.encoder.layer):
                modules_to_hook.append((f"layer_{i}", layer))
        # Mamba (HF implementation usually has 'backbone' or 'layers')
        # Structure: model.backbone.layers[i] or model.layers[i]
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layers'):
            for i, layer in enumerate(self.model.backbone.layers):
                modules_to_hook.append((f"layer_{i}", layer))
        elif hasattr(self.model, 'layers'): # Some Mamba implementations or Llama
             for i, layer in enumerate(self.model.layers):
                modules_to_hook.append((f"layer_{i}", layer))
        
        if not modules_to_hook:
            print("Warning: Could not auto-detect layers. No hooks registered.")
        
        for name, module in modules_to_hook:
            hook = module.register_forward_hook(self._get_hook_fn(name))
            self.hooks.append(hook)
            print(f"Registered hook on: {name}")

    def _get_hook_fn(self, name):
        def hook(module, input, output):
            # Transformer layer output is usually a tuple (hidden_state, present_key_value, ...)
            # We want the first element: hidden_state
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            
            # Store on CPU to save GPU memory
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(hidden_state.detach().cpu())
        return hook

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(self):
        return self.activations
