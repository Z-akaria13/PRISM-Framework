import numpy as np
from typing import Dict, List, Any

from prism.core.provenance import array_entries, weights_digest
from prism.inspectors.base import Inspector


class KerasInspector(Inspector):
    def __init__(self, model: Any):
        super().__init__(model)
        self.target_layers = []
        self.original_call = None
        self.feature_extractor = None

    def discover_topology(self) -> List[Dict[str, Any]]:
        """Returns the layers of the Keras model for the Wiretap tab."""
        layers = []
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                try:
                    params = f"{layer.count_params():,}"
                except Exception:  # an unbuilt layer has no parameter count yet
                    params = "0"
                layers.append({
                    "id": layer.name,
                    "type": layer.__class__.__name__,
                    "params": params
                })
        return layers

    def attach_hooks(self, layer_names: List[str]):
        """Creates a sub-model to extract features and intercepts the forward pass."""
        import tensorflow as tf
        self.teardown()
        self.target_layers = layer_names
        
        if not layer_names: return

        # 1. Create a Keras Sub-Model that outputs the target layers
        outputs = [self.model.get_layer(name).output for name in layer_names]
        self.feature_extractor = tf.keras.Model(inputs=self.model.inputs, outputs=outputs)

        # 2. Intercept the model's call method
        self.original_call = self.model.__call__

        def hooked_call(inputs, *args, **kwargs):
            # Run the sub-model to get hidden features
            activations = self.feature_extractor(inputs, training=False)
            if not isinstance(activations, list):
                activations = [activations]
            
            # Store as numpy arrays and fix Channel ordering
            for name, act in zip(self.target_layers, activations):
                arr = act.numpy()
                
                # TF is (Batch, H, W, Channels). 
                # PRISM expects (Batch, Channels, H, W) for CNN pooling math.
                if len(arr.shape) == 4:
                    arr = np.transpose(arr, (0, 3, 1, 2))
                    
                if name not in self._buffer: self._buffer[name] = []
                self._buffer[name].append(arr)

            # Run the original model to get the final predictions
            return self.original_call(inputs, *args, **kwargs)

        self.model.__call__ = hooked_call

    def model_fingerprint(self):
        """The weights in the order Keras lists them."""
        weights = self.model.get_weights()
        return weights_digest(array_entries((f"{i:06d}", w) for i, w in enumerate(weights)))

    def teardown(self):
        """Removes the interception so the user's model goes back to normal."""
        if self.original_call is not None:
            self.model.__call__ = self.original_call
            self.original_call = None
        self.feature_extractor = None
        self.target_layers = []
        self._buffer = {}