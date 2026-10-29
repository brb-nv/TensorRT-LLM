# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Install the booked TP8 locality factory and its INFO workspace receipt.

The pre-synth dual-residency check belongs to the measurement job. This hook
adds no traffic admission or TP2 locality lifecycle.
"""

from . import m31_local_moe_op as local_module
from .serving_hook import install


def install_tp8(model, mapping):
    if mapping.tp_size != 8 or mapping.tp_rank not in range(8):
        raise ValueError("TP8 locality bootstrap requires TP8 rank")
    previous = getattr(model, "_tp8_locality_bootstrap", None)
    identity = (mapping.tp_size, mapping.tp_rank)
    if previous is not None:
        if previous != identity:
            raise RuntimeError("TP8 locality bootstrap rank changed within process")
        return
    stock = model.M3MoeState

    # Same shape selection as the booked 883485 factory; make_state owns kill.
    def make_state(device, num_tokens, num_ctas=None, shared=False, **kwargs):
        if shared and num_tokens == 8:
            return local_module.make_state(
                device, num_tokens, num_ctas=num_ctas, shared=True, **kwargs
            )
        return stock(device, num_tokens=num_tokens, num_ctas=num_ctas, shared=shared, **kwargs)

    install(model, rank=mapping.tp_rank, make_state=make_state)
    model._tp8_locality_bootstrap = identity
