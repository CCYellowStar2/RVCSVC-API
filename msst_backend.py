import sys
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import librosa
import soundfile as sf
import torch


MSST_ROOT = Path(__file__).resolve().parent / "msst"
MSST_SELECTION = "MSST BS-Roformer"
MSST_MODEL_URL = (
    "https://modelscope.cn/models/CCYellowStar/bs_roformer_karaoke_frazer_becruily/"
    "resolve/master/bs_roformer_karaoke_frazer_becruily.ckpt"
)
MODEL_PROFILES = {
    MSST_SELECTION: (
        Path("pretrain/vocal_models/bs_roformer_karaoke_frazer_becruily.ckpt"),
        Path("configs/vocal_models/config_karaoke_frazer_becruily.yaml"),
    ),
}

if str(MSST_ROOT) not in sys.path:
    sys.path.insert(0, str(MSST_ROOT))

from inference.msst_infer import MSSeparator


_SEPARATOR_CACHE: dict[tuple[Path, str, str], MSSeparator] = {}
_SEPARATOR_LOCK = threading.RLock()


def asset_paths(root: Path = MSST_ROOT, model_selection: str = MSST_SELECTION) -> tuple[Path, Path]:
    root = Path(root)
    try:
        model_relative_path, config_relative_path = MODEL_PROFILES[model_selection]
    except KeyError as error:
        raise ValueError(f"Unknown MSST model selection: {model_selection}") from error
    return root / model_relative_path, root / config_relative_path


def ensure_model_downloaded(model_path: Path) -> Path:
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path

    temporary_path = model_path.with_suffix(model_path.suffix + ".part")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.unlink(missing_ok=True)
    print(f"Downloading missing MSST model to: {model_path}")
    try:
        with urlopen(MSST_MODEL_URL, timeout=60) as response, temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary_path.stat().st_size == 0:
            raise RuntimeError("download produced an empty file")
        temporary_path.replace(model_path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download MSST model to {model_path}: {error}") from error
    return model_path


def validate_assets(root: Path = MSST_ROOT, model_selection: str = MSST_SELECTION) -> tuple[Path, Path]:
    model_path, config_path = asset_paths(root, model_selection)
    model_missing = not model_path.is_file() or model_path.stat().st_size == 0
    if model_missing and config_path.is_file():
        ensure_model_downloaded(model_path)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"MSST model file not found: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"MSST config file not found: {config_path}")
    return model_path, config_path


def get_separator(root: Path, model_selection: str, device: str, output_dir: Path) -> MSSeparator:
    root = Path(root).resolve()
    cache_key = (root, model_selection, device)
    with _SEPARATOR_LOCK:
        separator = _SEPARATOR_CACHE.get(cache_key)
        if separator is not None:
            return separator

        model_path, config_path = validate_assets(root, model_selection)
        separator = MSSeparator(
            model_type="bs_roformer",
            config_path=str(config_path),
            model_path=str(model_path),
            device=device,
            output_format="wav",
            store_dirs=str(output_dir),
            inference_params={"batch_size": None, "num_overlap": None, "chunk_size": None, "normalize": None},
        )
        _SEPARATOR_CACHE[cache_key] = separator
        return separator


def separate_msst_audio(
    input_path: str | Path,
    output_dir: str | Path,
    root: Path = MSST_ROOT,
    model_selection: str = MSST_SELECTION,
) -> tuple[str, str]:
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"MSST input audio not found: {input_path}")

    root = Path(root).resolve()
    validate_assets(root, model_selection)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    separator = get_separator(root, model_selection, device, output_dir)
    mix, sample_rate = librosa.load(str(input_path), sr=44100, mono=False)
    with _SEPARATOR_LOCK:
        started_at = time.perf_counter()
        results = separator.separate(mix)
    print(f"MSST separation completed in {time.perf_counter() - started_at:.1f}s")
    normalized_results = {str(stem).lower(): audio for stem, audio in results.items()}
    vocal_audio = normalized_results.get("vocals")
    instrumental_audio = normalized_results.get("other")
    if instrumental_audio is None:
        instrumental_audio = normalized_results.get("instrumental")
    if vocal_audio is None or instrumental_audio is None:
        raise RuntimeError("MSST separation did not produce both vocal and instrumental files")

    vocal_path = output_dir / f"{input_path.stem}_vocals.wav"
    instrumental_path = output_dir / f"{input_path.stem}_other.wav"
    sf.write(vocal_path, vocal_audio, sample_rate, subtype="FLOAT")
    sf.write(instrumental_path, instrumental_audio, sample_rate, subtype="FLOAT")
    return str(vocal_path), str(instrumental_path)
