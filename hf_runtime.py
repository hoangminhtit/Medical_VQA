import os


def configure_hf_runtime(args=None):
    """Configure Hugging Face runtime behavior for non-token workflows.

    Supports:
    - shared cache directory
    - offline mode
    - request timeout tuning
    - warning/log verbosity control
    """
    hf_cache_dir = getattr(args, "hf_cache_dir", None) if args is not None else None
    hf_offline = bool(getattr(args, "hf_offline", False)) if args is not None else False
    hf_timeout = int(getattr(args, "hf_timeout", 120)) if args is not None else 120
    show_hf_warnings = bool(getattr(args, "show_hf_warnings", False)) if args is not None else False

    # Keep runtime clean unless explicitly requested.
    if not show_hf_warnings:
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        try:
            from huggingface_hub.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
        except Exception:
            pass

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Avoid very long silent waits on poor networks.
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(hf_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(hf_timeout)

    if hf_cache_dir:
        cache_root = os.path.abspath(hf_cache_dir)
        os.makedirs(cache_root, exist_ok=True)

        os.environ["HF_HOME"] = cache_root
        os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_root, "transformers")
        os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets")

    if hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
