from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import torch

if TYPE_CHECKING:
    import spandrel

logger = logging.getLogger(__name__)


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
    hash_prefix: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress, hash_prefix=hash_prefix)
    return cached_file


def load_models(model_path: str, model_url: str | None = None, download_name: str | None = None, ext_filter: list[str] | None = None, name_filter: str | None = None) -> list[str]:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param model_path: The location to store/find models in.
    @param model_url: If no other models are found, this will be downloaded.
    @param download_name: Specify to download from model_url immediately.
    @param ext_filter: An optional list of filename extensions to filter by
    @param name_filter: An optional string to filter by name
    @return: A list of paths containing the desired model(s)
    """
    output = []

    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for file in os.listdir(model_path):
            if ext_filter is not None:
                if not any(file.endswith(x) for x in ext_filter):
                    continue
            
            if name_filter is not None:
                if name_filter not in file:
                    continue

            full_path = os.path.join(model_path, file)
            if full_path not in output:
                output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                output.append(load_file_from_url(model_url, model_dir=model_path, file_name=download_name))
            else:
                output.append(model_url)

    except Exception:
        pass

    return output


def friendly_name(file: str):
    if file.startswith("http"):
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, extension = os.path.splitext(file)
    return model_name

# None: not loaded, False: failed to load, True: loaded
_spandrel_extra_init_state = None


def _init_spandrel_extra_archs() -> None:
    """
    Try to initialize `spandrel_extra_archs` (exactly once).
    """
    global _spandrel_extra_init_state
    if _spandrel_extra_init_state is not None:
        return

    try:
        import spandrel
        import spandrel_extra_arches
        spandrel.MAIN_REGISTRY.add(*spandrel_extra_arches.EXTRA_REGISTRY)
        _spandrel_extra_init_state = True
    except Exception:
        logger.warning("Failed to load spandrel_extra_arches", exc_info=True)
        _spandrel_extra_init_state = False


def load_spandrel_model(
    path: str | os.PathLike,
    *,
    device: str | torch.device | None,
    expected_architecture: str | None = None,
) -> spandrel.ModelDescriptor:
    global _spandrel_extra_init_state

    import spandrel
    _init_spandrel_extra_archs()

    model_descriptor = spandrel.ModelLoader(device=device).load_from_file(str(path))
    arch = model_descriptor.architecture
    if expected_architecture and arch.name != expected_architecture:
        logger.warning(
            f"Model {path!r} is not a {expected_architecture!r} model (got {arch.name!r})",
        )

    model_descriptor.model.eval()
    logger.debug(
        "Loaded %s from %s (device=%s)",
        arch, path, device,
    )
    return model_descriptor