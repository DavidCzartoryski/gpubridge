"""Registers ``backend="gpubridge"`` with torch.distributed when torch is imported.

gpubridge's ``torch.backends`` entry point points here, so that a script
needs no ``import gpubridge`` to call ``init_process_group(backend="gpubridge")``.
gpubridge itself is imported only when such a group is created. This must
never raise: an exception here would make ``import torch`` fail.
``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` turns all such hooks off; ``import gpubridge``
registers the name too.
"""


def _create(options, backend_options=None):
    from gpubridge.torch_backend import create

    return create(options, backend_options)


def register() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and "gpubridge" not in dist.Backend.backend_list:
            dist.Backend.register_backend("gpubridge", _create, extended_api=True,
                                          devices=["cpu", "cuda"])
    except Exception:  # noqa: BLE001, S110 - never break `import torch`
        pass
