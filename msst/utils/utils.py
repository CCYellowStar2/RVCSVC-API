import os
import numpy as np
import torch
import torch.nn as nn
import yaml
import librosa
import torch.nn.functional as F
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from numpy.typing import NDArray
from typing import Dict

from utils.logger import get_logger
logger = get_logger()


def _is_zluda_env():
    try:
        import zluda
        _running = getattr(zluda, 'runing', False)
        if _running:
            return True
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0).lower()
            if "amd" in name or "radeon" in name or "gfx" in name:
                return True
        except Exception:
            pass
    return False


def get_model_from_config(model_type, config_path):
    with open(config_path) as f:
        if model_type == 'htdemucs':
            config = OmegaConf.load(config_path)
        else:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    if model_type == 'mdx23c':
        from modules.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        from modules.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        from modules.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'torchseg':
        from modules.torchseg_models import Torchseg_Net
        model = Torchseg_Net(config)
    elif model_type == 'mel_band_roformer':
        from modules.bs_roformer import MelBandRoformer
        model = MelBandRoformer(
            **dict(config.model)
        )
    elif model_type == 'bs_roformer':
        from modules.bs_roformer import BSRoformer
        model = BSRoformer(
            **dict(config.model)
        )
    else:
        logger.error('Unknown model: {}'.format(model_type))
        model = None

    return model, config


def _getWindowingArray(window_size, fade_size):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window


def demix_track(config, model, mix, device, pbar=False):
    C = config.audio.chunk_size
    N = config.inference.num_overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = config.inference.batch_size

    length_init = mix.shape[-1]

    if length_init > 2 * border and (border > 0):
        if mix.ndim == 1:
            mix = mix.unsqueeze(0)
        mix = nn.functional.pad(mix, (border, border), mode='reflect')

    windowingArray = _getWindowingArray(C, fade_size)

    use_amp = config.training.get('use_amp', True)
    if device == 'cpu':
        use_amp = False
    is_zluda = _is_zluda_env()
    if is_zluda:
        use_amp = False
        logger.info("[MSST demix] ZLUDA detected: AMP disabled, using FP32 input (reflection_pad1d not supported in FP16)")

    _is_dml = False
    try:
        import torch_directml
        if str(device).startswith("privateuseone"):
            _is_dml = True
            use_amp = False
            logger.info("[MSST demix] DirectML detected: AMP disabled, using FP32")
    except ImportError:
        pass

    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    if _is_dml:
        _autocast_ctx = torch.inference_mode()
    else:
        _autocast_ctx = torch.amp.autocast(device_type, enabled=use_amp)

    with _autocast_ctx:
        with torch.inference_mode():
            if config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None

            while i < mix.shape[1]:
                part = mix[:, i:i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                    else:
                        part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    window = windowingArray.clone()
                    if i - step == 0:
                        window[:fade_size] = 1
                    elif i >= mix.shape[1]:
                        window[-fade_size:] = 1

                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start+l] += x[j][..., :l].cpu() * window[..., :l]
                        counter[..., start:start+l] += window[..., :l]

                    batch_data = []
                    batch_locations = []

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if length_init > 2 * border and (border > 0):
                estimated_sources = estimated_sources[..., border:-border]

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}


def demix(config, model, mix: NDArray, device, pbar=False, model_type: str = None) -> Dict[str, NDArray]:
    mix = torch.tensor(mix, dtype=torch.float32)
    return demix_track(config, model, mix, device, pbar=pbar)
