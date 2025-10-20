import ast
import base64
import datetime
import glob
import json
import logging
import multiprocessing
import os
import re
import requests
import shutil
import subprocess
import sys
import traceback
import zipfile
from itertools import chain
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

from compress_model import removeOptimizer
from fap.utils.file import AUDIO_EXTENSIONS, list_files
from fap.utils.slice_audio_v2 import slice_audio_file_v2
from fap.cli.length import length
from inference.infer_tool_webui import Svc
from onnx_export import main as onnx_export
from sami import SAMIService
from tts_voices import SUPPORTED_LANGUAGES
from utils import mix_model

os.environ["PATH"] += os.pathsep + os.path.join(os.getcwd(), "ffmpeg", "bin")

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Some directories
workdir = "logs/44k"
second_dir = "models"
diff_second_dir = "models/diffusion"
diff_workdir = "logs/44k/diffusion"
config_dir = "configs/"
dataset_dir = "dataset/44k"
raw_path = "dataset_raw"
raw_wavs_path = "raw"
models_backup_path = 'models_backup'
root_dir = "checkpoints"
default_settings_file = "settings.yaml"
current_mode = ""
# Some global variables
debug = False
precheck_ok = False
model = None
sovits_params = {}
diff_params = {}
# Some dicts for mapping
MODEL_TYPE = {
    "vec768l12": 768,
    "vec256l9": 256,
    "hubertsoft": 256,
    "whisper-ppg": 1024,
    "cnhubertlarge": 1024,
    "dphubert": 768,
    "wavlmbase+": 768,
    "whisper-ppg-large": 1280
}
ENCODER_PRETRAIN = {
    "vec256l9": "pretrain/checkpoint_best_legacy_500.pt",
    "vec768l12": "pretrain/checkpoint_best_legacy_500.pt",
    "hubertsoft": "pretrain/hubert-soft-0d54a1f4.pt",
    "whisper-ppg": "pretrain/medium.pt",
    "cnhubertlarge": "pretrain/chinese-hubert-large-fairseq-ckpt.pt",
    "dphubert": "pretrain/DPHuBERT-sp0.75.pth",
    "wavlmbase+": "pretrain/WavLM-Base+.pt",
    "whisper-ppg-large": "pretrain/large-v2.pt"
}


class Config:
    def __init__(self, path, type):
        self.path = path
        self.type = type
    
    def read(self):
        if self.type == "json":
            with open(self.path, 'r') as f:
                return json.load(f)
        if self.type == "yaml":
            with open(self.path, 'r') as f:
                return yaml.safe_load(f)
    
    def save(self, content):
        if self.type == "json":
            with open(self.path, 'w') as f:
                json.dump(content, f, indent=4)
        if self.type == "yaml":
            with open(self.path, 'w') as f:
                yaml.safe_dump(content, f, default_flow_style=False, sort_keys=False)


class ReleasePacker:
    def __init__(self, speaker, model):
        self.speaker = speaker
        self.model = model
        self.output_path = os.path.join("release_packs", f"{speaker}_release.zip")
        self.file_list = []

    def remove_temp(self, path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) and not filename.endswith(".zip"):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=True)

    def add_file(self, file_paths):
        self.file_list.extend(file_paths)
    
    def spk_to_dict(self):
        spk_string = self.speaker.replace('，', ',')
        spk_string = spk_string.replace(' ', '')
        _spk = spk_string.split(',')
        return {_spk: index for index, _spk in enumerate(_spk)}

    def generate_config(self, diff_model, config_origin):
        _config_origin = Config(os.path.join(config_read_dir, config_origin), "json")
        _template = Config("release_packs/config_template.json", "json")
        _d_template = Config("release_packs/diffusion_template.yaml", "yaml")
        orig_config = _config_origin.read()
        config_template = _template.read()
        diff_config_template = _d_template.read()
        spk_dict = self.spk_to_dict()
        _net = torch.load(os.path.join(ckpt_read_dir, self.model), map_location='cpu')
        emb_dim, model_dim = _net['model'].get('emb_g.weight', torch.empty(0, 0)).size()
        vol_emb = _net['model'].get('emb_vol.weight')
        if vol_emb is not None:
            config_template["train"]["vol_aug"] = config_template["model"]["vol_embedding"] = True
        #Keep the spk_dict length same as emb_dim
        if emb_dim > len(spk_dict):
            for i in range(emb_dim - len(spk_dict)):
                spk_dict[f"spk{i}"] = len(spk_dict)
        if emb_dim < len(spk_dict):
            for i in range(len(spk_dict) - emb_dim):
                spk_dict.popitem()
        self.speaker = ','.join(spk_dict.keys())
        config_template['model']['ssl_dim'] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = model_dim
        config_template['model']['n_speakers'] = diff_config_template['model']['n_spk'] = emb_dim
        config_template['spk'] = diff_config_template['spk'] = spk_dict
        encoder = [k for k, v in MODEL_TYPE.items() if v == model_dim]
        if orig_config['model']['speech_encoder'] in encoder:
            config_template['model']['speech_encoder'] = orig_config['model']['speech_encoder']
        else:
            raise Exception("Config is not compatible with the model")
        
        if diff_model != "no_diff":
            _diff = torch.load(os.path.join(diff_read_dir, diff_model), map_location='cpu')
            _, diff_dim = _diff["model"].get("unit_embed.weight", torch.empty(0, 0)).size()
            if diff_dim == 256:
                diff_config_template['data']['encoder'] = 'hubertsoft'
                diff_config_template['data']['encoder_out_channels'] = 256
            elif diff_dim == 768:
                diff_config_template['data']['encoder'] = 'vec768l12'
                diff_config_template['data']['encoder_out_channels'] = 768
            elif diff_dim == 1024:
                diff_config_template['data']['encoder'] = 'whisper-ppg'
                diff_config_template['data']['encoder_out_channels'] = 1024

        with open("release_packs/install.txt", 'w') as f:
            f.write(str(self.file_list) + '#' + str(self.speaker))

        _template.save(config_template)
        _d_template.save(diff_config_template)

    def unpack(self, zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            zipf.extractall("release_packs")

    def formatted_install(self, install_txt):
        with open(install_txt, 'r') as f:
            content = f.read()
        file_list, speaker = content.split('#')
        self.speaker = speaker
        file_list = ast.literal_eval(file_list)
        self.file_list = file_list
        for _, target_path in self.file_list:
            if target_path != "install.txt" and target_path != "":
                shutil.move(os.path.join("release_packs", target_path), target_path)
        self.remove_temp("release_packs")
        return self.speaker

    def pack(self):
        with zipfile.ZipFile(self.output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, target_path in self.file_list:
                if os.path.isfile(file_path):
                    zipf.write(file_path, arcname=target_path)


def debug_change():
    global debug
    debug = debug_button.value

def get_default_settings():
    global sovits_params, diff_params, second_dir_enable
    config_file = Config(default_settings_file, "yaml")
    default_settings = config_file.read()
    sovits_params = default_settings['sovits_params']
    diff_params = default_settings['diff_params']
    webui_settings = default_settings['webui_settings']
    second_dir_enable = webui_settings['second_dir']
    sami_settings = default_settings['sami_settings']
    return sovits_params, diff_params, second_dir_enable, sami_settings

def webui_change(read_second_dir):
    global second_dir_enable
    config_file = Config(default_settings_file, "yaml")
    default_settings = config_file.read()
    second_dir_enable = default_settings['webui_settings']['second_dir'] = read_second_dir
    config_file.save(default_settings)

def get_current_mode():
    global current_mode
    current_mode = "当前模式：独立目录模式，将从'./models/'读取模型文件" if second_dir_enable else "当前模式：工作目录模式，将从'./logs/44k'读取模型文件" 
    return current_mode

def save_default_settings(log_interval,eval_interval,keep_ckpts,batch_size,learning_rate,amp_dtype,all_in_mem,num_workers,cache_all_data,cache_device,diff_amp_dtype,diff_batch_size,diff_lr,diff_interval_log,diff_interval_val,diff_force_save,diff_k_step_max):
    config_file = Config(default_settings_file, "yaml")
    default_settings = config_file.read()
    default_settings['sovits_params']['log_interval'] = int(log_interval)
    default_settings['sovits_params']['eval_interval'] = int(eval_interval)
    default_settings['sovits_params']['keep_ckpts'] = int(keep_ckpts)
    default_settings['sovits_params']['batch_size'] = int(batch_size)
    default_settings['sovits_params']['learning_rate'] = float(learning_rate)
    default_settings['sovits_params']['amp_dtype'] = str(amp_dtype)
    default_settings['sovits_params']['all_in_mem'] = all_in_mem
    default_settings['diff_params']['num_workers'] = int(num_workers)
    default_settings['diff_params']['cache_all_data'] = cache_all_data
    default_settings['diff_params']['cache_device'] = str(cache_device)
    default_settings['diff_params']['amp_dtype'] = str(diff_amp_dtype)
    default_settings['diff_params']['diff_batch_size'] = int(diff_batch_size)
    default_settings['diff_params']['diff_lr'] = float(diff_lr)
    default_settings['diff_params']['diff_interval_log'] = int(diff_interval_log)
    default_settings['diff_params']['diff_interval_val'] = int(diff_interval_val)
    default_settings['diff_params']['diff_force_save'] = int(diff_force_save)
    default_settings['diff_params']['diff_k_step_max'] = diff_k_step_max
    config_file.save(default_settings)
    return "成功保存默认配置"

def get_model_info(choice_ckpt):
    pthfile = os.path.join(ckpt_read_dir, choice_ckpt)
    net = torch.load(pthfile, map_location=torch.device('cpu')) #cpu load to avoid using gpu memory
    spk_emb = net["model"].get("emb_g.weight")
    if spk_emb is None:
        return "所选模型缺少emb_g.weight，你可能选择了一个底模"
    _layer = spk_emb.size(1)
    encoder = [k for k, v in MODEL_TYPE.items() if v == _layer] #通过维度对应编码器
    encoder.sort()
    if encoder == ["hubertsoft", "vec256l9"]:
        encoder = ["vec256l9 / hubertsoft"]
    if encoder == ["cnhubertlarge", "whisper-ppg"]:
        encoder = ["whisper-ppg / cnhubertlarge"]
    if encoder == ["dphubert", "vec768l12", "wavlmbase+"]:
        encoder = ["vec768l12 / dphubert / wavlmbase+"]
    return encoder[0]
    
def load_json_encoder(config_choice, choice_ckpt):
    if config_choice == "no_config":
        return "未启用自动加载，请手动选择配置文件"
    if choice_ckpt == "no_model":
        return "请先选择模型"
    config_file = Config(os.path.join(config_read_dir, config_choice), "json")
    config = config_file.read()
    try:
        #比对配置文件中的模型维度与该encoder的实际维度是否对应，防止古神语
        config_encoder = config["model"].get("speech_encoder", "no_encoder")
        config_dim = config["model"]["ssl_dim"]
        #旧版配置文件自动匹配
        if config_encoder == "no_encoder":
            config_encoder = config["model"]["speech_encoder"] = "vec256l9" if config_dim == 256 else "vec768l12"
            config_file.save(config)
        correct_dim = MODEL_TYPE.get(config_encoder, "unknown")
        if config_dim != correct_dim:
            return "配置文件中的编码器与模型维度不匹配"
        return config_encoder
    except Exception as e:
        return f"出错了: {e}"
        
def auto_load(choice_ckpt):
    global second_dir_enable
    model_output_msg = get_model_info(choice_ckpt)
    json_output_msg = config_choice = ""
    choice_ckpt_name, _ = os.path.splitext(choice_ckpt)
    if second_dir_enable:
        all_config = [json for json in os.listdir(second_dir) if json.endswith(".json")]
        for config in all_config:
            config_fname, _ = os.path.splitext(config)
            if config_fname == choice_ckpt_name:
                config_choice = config
                json_output_msg = load_json_encoder(config, choice_ckpt)
        if json_output_msg != "":
            return model_output_msg, config_choice, json_output_msg
        else:
            return model_output_msg, "no_config", ""
    else:
        return model_output_msg, "no_config", ""
    
def auto_load_diff(diff_model):
    global second_dir_enable
    if second_dir_enable is False:
        return "no_diff_config"
    all_diff_config = [yaml for yaml in os.listdir(second_dir) if yaml.endswith(".yaml")]
    for config in all_diff_config:
        config_fname, _ = os.path.splitext(config)
        diff_fname, _ = os.path.splitext(diff_model)
        if config_fname == diff_fname:
            return config
    return "no_diff_config"
        
def load_model_func(ckpt_name,cluster_name,config_name,enhance,diff_model_name,diff_config_name,only_diffusion,use_spk_mix,using_device,method,speedup,cl_num,vocoder_name):
    global model
    config_path = os.path.join(config_read_dir, config_name) if not only_diffusion else "configs/config.json"
    diff_config_path = os.path.join(config_read_dir, diff_config_name) if diff_config_name != "no_diff_config" else "configs/diffusion.yaml"
    ckpt_path = os.path.join(ckpt_read_dir, ckpt_name)
    cluster_path = os.path.join(ckpt_read_dir, cluster_name)
    diff_model_path = os.path.join(diff_read_dir, diff_model_name)
    k_step_max = 1000
    if not only_diffusion:
        config = Config(config_path, "json").read()
    if diff_model_name != "no_diff":
        _diff = Config(diff_config_path, "yaml")
        _content = _diff.read()
        diff_spk = _content.get('spk', {})
        diff_spk_choice = spk_choice = next(iter(diff_spk), "未检测到音色")
        if not only_diffusion:
            if _content['data'].get('encoder_out_channels') != config["model"].get('ssl_dim'):
                return "扩散模型维度与主模型不匹配，请确保两个模型使用的是同一个编码器", gr.Dropdown.update(choices=[], value=""), 0, None
        _content["infer"]["speedup"] = int(speedup)
        _content["infer"]["method"] = str(method)
        _content["vocoder"]["ckpt"] = f"pretrain/{vocoder_name}/model"
        k_step_max = _content["model"].get('k_step_max', 0) if _content["model"].get('k_step_max', 0) != 0 else 1000
        _diff.save(_content)
    if not only_diffusion:
        net = torch.load(ckpt_path, map_location=torch.device('cpu'))
    #读取模型各维度并比对，还有小可爱无视提示硬要加载底模的就返回个未初始张量
        emb_dim, model_dim = net["model"].get("emb_g.weight", torch.empty(0, 0)).size() 
        if emb_dim > config["model"]["n_speakers"]:
            return "模型说话人数量与emb维度不匹配", gr.Dropdown.update(choices=[], value=""), 0, None
        if model_dim != config["model"]["ssl_dim"]: 
            return "配置文件与模型不匹配", gr.Dropdown.update(choices=[], value=""), 0, None
        encoder = config["model"]["speech_encoder"]
        spk_dict = config.get('spk', {})
        spk_choice = next(iter(spk_dict), "未检测到音色")
    else:
        spk_dict = diff_spk
        spk_choice = diff_spk_choice
    fr = cluster_name.endswith(".pkl") #如果是pkl后缀就启用特征检索
    shallow_diffusion = diff_model_name != "no_diff" #加载了扩散模型就启用浅扩散
    device = cuda[using_device] if "CUDA" in using_device else using_device
    model = Svc(ckpt_path,
                    config_path,
                    device=device if device != "Auto" else None,
                    cluster_model_path=cluster_path,
                    nsf_hifigan_enhance=enhance,
                    diffusion_model_path=diff_model_path,
                    diffusion_config_path=diff_config_path,
                    shallow_diffusion=shallow_diffusion,
                    only_diffusion=only_diffusion,
                    spk_mix_enable=use_spk_mix,
                    feature_retrieval=fr)
    spk_list = list(spk_dict.keys())
    if enhance:
        from modules.enhancer import Enhancer
        model.enhancer = Enhancer('nsf-hifigan', f'pretrain/{vocoder_name}/model', device=model.dev)
    if not only_diffusion:
        clip = 25 if encoder == "whisper-ppg" or encoder == "whisper-ppg-large" else cl_num #Whisper必须强制切片25秒
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        sovits_msg = f"模型被成功加载到了{device_name}上\n"
    else: 
        clip = cl_num
        sovits_msg = "启用全扩散推理，未加载So-VITS模型\n"
    index_or_kmeans = "特征索引" if fr else "聚类模型"
    clu_load = "未加载" if cluster_name == "no_clu" else cluster_name
    diff_load = "未加载" if diff_model_name == "no_diff" else f"{diff_model_name} | 采样器: {method} | 加速倍数：{int(speedup)} | 最大浅扩散步数：{k_step_max} | 声码器： {vocoder_name}"
    output_msg = f"{sovits_msg}{index_or_kmeans}：{clu_load}\n扩散模型：{diff_load}"
    return (
        output_msg, 
        gr.Dropdown.update(choices=spk_list, value=spk_choice), 
        clip, 
        gr.Slider.update(value=100 if k_step_max>100 else k_step_max, minimum=speedup, maximum=k_step_max)
    )

def model_empty_cache():
    global model
    if model is None:
        return sid.update(choices = [],value=""),"没有模型需要卸载!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices = [],value=""),"模型卸载完毕!"

def get_file_options(directory, extension):
    return [file for file in os.listdir(directory) if file.endswith(extension)]

def load_options():
    ckpt_list = [file for file in get_file_options(ckpt_read_dir, ".pth") if not file.startswith("D_") or file == "G_0.pth"]
    config_list = get_file_options(config_read_dir, ".json")
    cluster_list = ["no_clu"] + get_file_options(ckpt_read_dir, ".pt") + get_file_options(ckpt_read_dir, ".pkl") # 聚类和特征检索模型
    diff_list = ["no_diff"] + get_file_options(diff_read_dir, ".pt")
    diff_config_list = ["no_diff_config"] + get_file_options(config_read_dir, ".yaml")
    return ckpt_list, config_list, cluster_list, diff_list, diff_config_list

def refresh_options():
    global ckpt_read_dir, config_read_dir, diff_read_dir, current_mode
    ckpt_read_dir = second_dir if second_dir_enable else workdir
    config_read_dir = second_dir if second_dir_enable else config_dir
    diff_read_dir = diff_second_dir if second_dir_enable else diff_workdir
    ckpt_list, config_list, cluster_list, diff_list, diff_config_list = load_options()
    current_mode = get_current_mode()
    return (
        choice_ckpt.update(choices=ckpt_list),
        config_choice.update(choices=config_list),
        cluster_choice.update(choices=cluster_list),
        diff_choice.update(choices=diff_list),
        diff_config_choice.update(choices=diff_config_list),
        mode_caption.update(value=f"""{current_mode}，可在页面底端切换模式""")
    )
    
def refresh_options2():
    global ckpt_read_dir, config_read_dir, diff_read_dir, current_mode
    ckpt_read_dir = second_dir if second_dir_enable else workdir
    config_read_dir = second_dir if second_dir_enable else config_dir
    diff_read_dir = diff_second_dir if second_dir_enable else diff_workdir
    ckpt_list, config_list, cluster_list, diff_list, diff_config_list = load_options()
    current_mode = get_current_mode()
    return (
        ckpt_list,
        config_list,
        cluster_list,
        diff_list,
        diff_config_list,
        f"""{current_mode}，可在页面底端切换模式"""
    )

def source_change(use_microphone):
    if use_microphone:
        return vc_input3.update(source="microphone")
    else:
        return vc_input3.update(source="upload")

def vc_infer(output_format, sid, input_audio, sr, input_audio_path, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    if np.issubdtype(input_audio.dtype, np.integer):
        input_audio = (input_audio / np.iinfo(input_audio.dtype).max).astype(np.float32)
    if len(input_audio.shape) > 1:
        input_audio = librosa.to_mono(input_audio.transpose(1, 0))
    if sr != 44100:
        input_audio = librosa.resample(input_audio, orig_sr=sr, target_sr=44100)
    sf.write("temp.wav", input_audio, 44100, format="wav")
    _audio = model.slice_inference(
        "temp.wav",
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )  
    model.clear_empty()
    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits_"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff_"
    if model.only_diffusion:
        isdiffusion = "diff_"
    #Gradio上传的filepath因为未知原因会有一个无意义的固定后缀，这里去掉
    truncated_basename = Path(input_audio_path).stem[:-6] if Path(input_audio_path).stem[-6:] == "-0-100" else Path(input_audio_path).stem
    output_file_name = f'{truncated_basename}_{sid}_{key}{cluster}{isdiffusion}{f0_predictor}.{output_format}'
    output_file_path = os.path.join("results", output_file_name)
    if os.path.exists(output_file_path):
        count = 1
        while os.path.exists(output_file_path):
            output_file_name = f'{truncated_basename}_{sid}_{key}{cluster}{isdiffusion}{f0_predictor}_{str(count)}.{output_format}'
            output_file_path = os.path.join("results", output_file_name)
            count += 1  
    sf.write(output_file_path, _audio, model.target_sample, format=output_format)
    return output_file_path

def vc_fn(output_format, sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment, progress=gr.Progress(track_tqdm=True)):
    global model
    try:
        if input_audio is None:
            return "你还没有上传音频", None
        if model is None:
            return "你还没有加载模型", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "你还未加载聚类或特征检索模型，无法启用聚类/特征检索混合比例", None
        audio, sr = sf.read(input_audio)
        output_file_path = vc_infer(output_format, sid, audio, sr, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        os.remove("temp.wav")
        return "Success", output_file_path
    except torch.cuda.OutOfMemoryError as e:
        raise gr.Error(f"{e}\n显存不足，请参考文档解决：https://www.yuque.com/umoubuton/ueupp5/ieinf8qmpzswpsvr")
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def vc_batch_fn(output_format, sid, input_audio_files, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment, progress=gr.Progress()):
    global model
    try:
        if input_audio_files is None or len(input_audio_files) == 0:
            return "你还没有上传音频"
        if model is None:
            return "你还没有加载模型"
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "你还未加载聚类或特征检索模型，无法启用聚类/特征检索混合比例", None
        _output = []
        for file_obj in progress.tqdm(input_audio_files, desc="Inferencing"):
            print(f"Start processing: {file_obj.name}")
            input_audio_path = file_obj.name
            audio, sr = sf.read(input_audio_path)
            output_file_path = vc_infer(output_format, sid, audio, sr, input_audio_path, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
            _output.append(output_file_path)
        return "批量推理完成，音频已经被保存到results文件夹"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)
    
def tts_fn(_text, _gender, _lang, _rate, _volume, output_format, sid, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold, k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment,progress=gr.Progress(track_tqdm=True)):
    global model
    try:
        if model is None:
            return "你还没有加载模型", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "你还未加载聚类或特征检索模型，无法启用聚类/特征检索混合比例", None
        _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
        _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
        if _lang == "Auto":
            _gender = "Male" if _gender == "男" else "Female"
            subprocess.run([r".\workenv\python.exe", "tts.py", _text, _lang, _rate, _volume, _gender])
        else:
            subprocess.run([r".\workenv\python.exe", "tts.py", _text, _lang, _rate, _volume])
        target_sr = 44100
        y, sr = librosa.load("tts.wav")
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sf.write("tts.wav", resampled_y, target_sr, subtype = "PCM_16")
        input_audio = "tts.wav"
        audio, sr = sf.read(input_audio)
        output_file_path = vc_infer(output_format, sid, audio, sr, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        #os.remove("tts.wav")
        return "Success", output_file_path
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def load_raw_dirs():
    global precheck_ok
    precheck_ok = False
    allowed_pattern = re.compile(r'^[a-zA-Z0-9_@#$%^&()_+\-=\s\.]*$')
    illegal_files = illegal_dataset = []
    for root, dirs, files in os.walk(raw_path):
        for dir in dirs:
            if not allowed_pattern.match(dir):
                illegal_dataset.append(dir)
        if illegal_dataset:
            return f"数据集文件夹名只能包含数字、字母、下划线，以下文件夹不符合要求，请改名后再试：\n{illegal_dataset}"
        if root != raw_path:  # 只处理子文件夹内的文件
            for file in files:
                if not allowed_pattern.match(file) and file not in illegal_files:
                    illegal_files.append(file)
                if not file.lower().endswith('.wav') and file not in illegal_files:
                    illegal_files.append(file)
    if illegal_files:
        return f"数据集文件名只能包含数字、字母、下划线，且必须是.wav格式，以下文件不符合要求，请改名后再试：\n{illegal_files}"
    spk_dirs = [entry.name for entry in os.scandir(raw_path) if entry.is_dir()]
    if spk_dirs:
        precheck_ok = True
        return spk_dirs
    else:
        return "未找到数据集，请检查dataset_raw文件夹"
    
def dataset_preprocess(encoder, f0_predictor, use_diff, vol_aug, skip_loudnorm, num_processes, tiny_enable):
    if precheck_ok:
        diff_arg = "--use_diff" if use_diff else ""
        vol_aug_arg = "--vol_aug" if vol_aug else ""
        skip_loudnorm_arg = "--skip_loudnorm" if skip_loudnorm else ""
        tiny_arg = "--tiny" if tiny_enable else ""
        preprocess_commands = [
            r".\workenv\python.exe resample.py %s" % (skip_loudnorm_arg),
            r".\workenv\python.exe preprocess_flist_config.py --speech_encoder %s %s %s" % (encoder, vol_aug_arg, tiny_arg),
            r".\workenv\python.exe preprocess_hubert_f0.py --num_processes %s --f0_predictor %s %s" % (num_processes ,f0_predictor, diff_arg)
            ]
        accumulated_output = ""
        #清空dataset
        dataset = os.listdir(dataset_dir)
        if len(dataset) != 0:
            for dir in dataset:
                dataset_spk_dir = os.path.join(dataset_dir, str(dir))
                if os.path.isdir(dataset_spk_dir):
                    shutil.rmtree(dataset_spk_dir)
                    accumulated_output += f"Deleting previous dataset: {dir}\n"
        for command in preprocess_commands:
            try:
                result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
                accumulated_output += f"Command: {command}, Using Encoder: {encoder}, Using f0 Predictor: {f0_predictor}\n"
                yield accumulated_output, None
                progress_line = None
                for line in result.stdout:
                    if r"it/s" in line or r"s/it" in line: #防止进度条刷屏
                        progress_line = line
                    else:
                        accumulated_output += line
                    if progress_line is None:
                        yield accumulated_output, None
                    else:
                        yield accumulated_output + progress_line, None
                result.communicate()
            except subprocess.CalledProcessError as e:
                result = e.output
                accumulated_output += f"Error: {result}\n"
                yield accumulated_output, None
            if progress_line is not None:
                accumulated_output += progress_line
            accumulated_output += '-' * 50 + '\n'
            yield accumulated_output, None
            config_path = "configs/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        spk_name = config.get('spk', None)
        yield accumulated_output, gr.Textbox.update(value=spk_name)
    else:
        yield "数据集识别未通过，请先识别数据集并确保没有报错信息", None

def regenerate_config(encoder, vol_aug, tiny_enable):
    if precheck_ok is False:
        return "数据集识别未通过，请检查识别结果的报错信息"
    vol_aug_arg = "--vol_aug" if vol_aug else ""
    tiny_arg = "--tiny" if tiny_enable else ""
    cmd = r".\workenv\python.exe preprocess_flist_config.py --speech_encoder %s %s %s" % (encoder, vol_aug_arg, tiny_arg)
    output = ""
    try:
        result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
        for line in result.stdout:
            output += line
        output += "Regenerate config file successfully."
    except subprocess.CalledProcessError as e:
        result = e.output
        output += f"Error: {result}\n"
    return output

def clear_output():
    return gr.Textbox.update(value="Cleared!>_<")

def get_available_encoder():
    current_pretrain = os.listdir("pretrain")
    current_pretrain = [("pretrain/" + model) for model in current_pretrain]
    encoder_list = []
    for encoder, path in ENCODER_PRETRAIN.items():
        if path in current_pretrain:
            encoder_list.append(encoder)
    return encoder_list

def config_fn(log_interval, eval_interval, keep_ckpts, batch_size, lr, amp_dtype, all_in_mem, diff_num_workers, diff_cache_all_data, diff_batch_size, diff_lr, diff_interval_log, diff_interval_val, diff_cache_device, diff_amp_dtype, diff_force_save, diff_k_step_max):
    if amp_dtype == "fp16" or amp_dtype == "bf16":
        fp16_run = True
    else:
        fp16_run = False
        amp_dtype = "fp16"
    config_origin = Config("configs/config.json", "json")
    diff_config = Config("configs/diffusion.yaml", "yaml")
    config_data = config_origin.read()
    config_data['train']['log_interval'] = int(log_interval)
    config_data['train']['eval_interval'] = int(eval_interval)
    config_data['train']['keep_ckpts'] = int(keep_ckpts)
    config_data['train']['batch_size'] = int(batch_size)
    config_data['train']['learning_rate'] = float(lr)
    config_data['train']['fp16_run'] = fp16_run
    config_data['train']['half_type'] = str(amp_dtype)
    config_data['train']['all_in_mem'] = all_in_mem
    config_origin.save(config_data)
    diff_config_data = diff_config.read()
    diff_config_data['train']['num_workers'] = int(diff_num_workers)
    diff_config_data['train']['cache_all_data'] = diff_cache_all_data
    diff_config_data['train']['batch_size'] = int(diff_batch_size)
    diff_config_data['train']['lr'] = float(diff_lr)
    diff_config_data['train']['interval_log'] = int(diff_interval_log)
    diff_config_data['train']['interval_val'] = int(diff_interval_val)
    diff_config_data['train']['cache_device'] = str(diff_cache_device)
    diff_config_data['train']['amp_dtype'] = str(diff_amp_dtype)
    diff_config_data['train']['interval_force_save'] = int(diff_force_save)
    diff_config_data['model']['k_step_max'] = 100 if diff_k_step_max else 0
    diff_config.save(diff_config_data)
    return "配置文件写入完成"

def check_dataset(dataset_path):
    if not os.listdir(dataset_path):
        return "数据集不存在，请检查dataset文件夹"
    no_npy_pt_files = True
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy') or file.endswith('.pt'):
                no_npy_pt_files = False
                break
    if no_npy_pt_files:
        return "数据集中未检测到f0和hubert文件，可能是预处理未完成"
    return None

def training(gpu_selection, encoder, tiny_enable):
    if tiny_enable:
        encoder = "vec768l12_tiny"
    config_file = Config("configs/config.json", "json")
    config_data = config_file.read()
    vol_emb = config_data["model"]["vol_embedding"]
    dataset_warn = check_dataset(dataset_dir)
    if dataset_warn is not None:
        return dataset_warn
    PRETRAIN = { 
        "vec256l9": ("D_0.pth", "G_0.pth", "pre_trained_model"),
        "vec768l12": ("D_0.pth", "G_0.pth", "pre_trained_model/768l12/vol_emb" if vol_emb else "pre_trained_model/768l12"),
        "vec768l12_tiny": ("D_0.pth", "G_0.pth", "pre_trained_model/tiny/vec768l12_vol_emb"),
        "hubertsoft": ("D_0.pth", "G_0.pth", "pre_trained_model/hubertsoft"),
        "whisper-ppg": ("D_0.pth", "G_0.pth", "pre_trained_model/whisper-ppg"),
        "cnhubertlarge": ("D_0.pth", "G_0.pth", "pre_trained_model/cnhubertlarge"),
        "dphubert": ("D_0.pth", "G_0.pth", "pre_trained_model/dphubert"),
        "wavlmbase+": ("D_0.pth", "G_0.pth", "pre_trained_model/wavlmbase+"),
        "whisper-ppg-large": ("D_0.pth", "G_0.pth", "pre_trained_model/whisper-ppg-large")
    }
    if encoder not in PRETRAIN:
        return "未知编码器"
    d_0_file, g_0_file, encoder_model_path = PRETRAIN[encoder]
    d_0_path = os.path.join(encoder_model_path, d_0_file)
    g_0_path = os.path.join(encoder_model_path, g_0_file)
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    new_backup_folder = os.path.join(models_backup_path, str(timestamp))
    output_msg = ""
    if os.listdir(workdir) != ['diffusion']:
        os.makedirs(new_backup_folder, exist_ok=True)
        for file in os.listdir(workdir):
            if file != "diffusion":
                shutil.move(os.path.join(workdir, file), os.path.join(new_backup_folder, file))
    if os.path.isfile(g_0_path) and os.path.isfile(d_0_path):
        shutil.copy(d_0_path, os.path.join(workdir, "D_0.pth"))
        shutil.copy(g_0_path, os.path.join(workdir, "G_0.pth"))
        output_msg += f"成功装载预训练模型，编码器：{encoder}\n"
    else:
        output_msg += f"{encoder}的预训练模型不存在，未装载预训练模型\n"

    cmd = r"set CUDA_VISIBLE_DEVICES=%s && .\workenv\python.exe train.py -c configs/config.json -m 44k" % (gpu_selection)
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd])
    output_msg += "已经在新的终端窗口开始训练，请监看终端窗口的训练日志。在终端中按Ctrl+C可暂停训练。"
    return output_msg

def continue_training(gpu_selection, encoder):
    dataset_warn = check_dataset(dataset_dir)
    if dataset_warn is not None:
        return dataset_warn
    if encoder == "":
        return "请先选择预处理对应的编码器"
    all_files = os.listdir(workdir)
    model_files = [f for f in all_files if f.startswith('G_') and f.endswith('.pth')]
    if len(model_files) == 0:
        return "你还没有已开始的训练"
    cmd = r"set CUDA_VISIBLE_DEVICES=%s && .\workenv\python.exe train.py -c configs/config.json -m 44k" % (gpu_selection)
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd])
    return "已经在新的终端窗口开始训练，请监看终端窗口的训练日志。在终端中按Ctrl+C可暂停训练。"

def kmeans_training(kmeans_gpu):
    if not os.listdir(dataset_dir):
        return "数据集不存在，请检查dataset文件夹"
    cmd = r".\workenv\python.exe cluster/train_cluster.py --gpu" if kmeans_gpu else r".\workenv\python.exe cluster/train_cluster.py"
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd])
    return "已经在新的终端窗口开始训练，训练聚类模型不会输出日志，CPU训练一般需要5-10分钟左右"

def index_training():
    if not os.listdir(dataset_dir):
        return "数据集不存在，请检查dataset文件夹"
    cmd = r".\workenv\python.exe train_index.py -c configs/config.json"
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd])
    return "已经在新的终端窗口开始训练"

def diff_training(encoder, k_step_max):
    if not os.listdir(dataset_dir):
        return "数据集不存在，请检查dataset文件夹"
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    new_backup_folder = os.path.join(models_backup_path, "diffusion", str(timestamp))
    if len(os.listdir(diff_workdir)) != 0:
        os.makedirs(new_backup_folder, exist_ok=True)
        for file in os.listdir(diff_workdir):
            shutil.move(os.path.join(diff_workdir, file), os.path.join(new_backup_folder, file))
    DIFF_PRETRAIN = {
        "768-kstepmax100": "pre_trained_model/diffusion/768l12/max100/model_0.pt",
        "vec768l12": "pre_trained_model/diffusion/768l12/model_0.pt",
        "hubertsoft": "pre_trained_model/diffusion/hubertsoft/model_0.pt",
        "whisper-ppg": "pre_trained_model/diffusion/whisper-ppg/model_0.pt"
    }
    if encoder not in DIFF_PRETRAIN:
        return "你所选的编码器暂时不支持训练扩散模型"
    if k_step_max:
        encoder = "768-kstepmax100"
    diff_pretrained_model = DIFF_PRETRAIN[encoder]
    shutil.copy(diff_pretrained_model, os.path.join(diff_workdir, "model_0.pt"))
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", r".\workenv\python.exe train_diff.py -c configs/diffusion.yaml"])
    output_message = "已经在新的终端窗口开始训练，请监看终端窗口的训练日志。在终端中按Ctrl+C可暂停训练。"
    if encoder == "768-kstepmax100":
        output_message += "\n正在进行100步深度的浅扩散训练，已加载底模"
    else:
        output_message += f"\n正在进行完整深度的扩散训练，编码器{encoder}"
    return output_message

def diff_continue_training(encoder):
    if not os.listdir(dataset_dir):
        return "数据集不存在，请检查dataset文件夹"
    if encoder == "":
        return "请先选择预处理对应的编码器"
    all_files = os.listdir(diff_workdir)
    model_files = [f for f in all_files if f.endswith('.pt')]
    if len(model_files) == 0:
        return "你还没有已开始的训练"
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", r".\workenv\python.exe train_diff.py -c configs/diffusion.yaml"])
    return "已经在新的终端窗口开始训练，请监看终端窗口的训练日志。在终端中按Ctrl+C可暂停训练。"

def upload_mix_append_file(files,sfiles):
    try:
        if(sfiles is None):
            file_paths = [file.name for file in files]
        else:
            file_paths = [file.name for file in chain(files,sfiles)]
        p = {file:100 for file in file_paths}
        return file_paths,mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def mix_submit_click(js,mode):
    try:
        assert js.lstrip()!=""
        modes = {"凸组合":0, "线性组合":1}
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path,mix_rate = zip(*data)
        path = mix_model(model_path,mix_rate,mode)
        return f"成功，文件被保存在了{path}"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def updata_mix_info(files):
    try:
        if files is None:
            return mix_model_output1.update(value="")
        p = {file.name:100 for file in files}
        return mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def pth_identify():
    if not os.path.exists(root_dir):
        return f"未找到{root_dir}文件夹，请先创建一个{root_dir}文件夹并按第一步流程操作"
    model_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if not model_dirs:
        return f"未在{root_dir}文件夹中找到模型文件夹，请确保每个模型和配置文件都被放置在单独的文件夹中"
    valid_model_dirs = []
    for path in model_dirs:
        pth_files = glob.glob(f"{root_dir}/{path}/*.pth")
        json_files = glob.glob(f"{root_dir}/{path}/*.json")
        if len(pth_files) != 1 or len(json_files) != 1:
            return f"错误: 在{root_dir}/{path}中找到了{len(pth_files)}个.pth文件和{len(json_files)}个.json文件。应当确保每个文件夹内有且只有一个.pth文件和.json文件"
        valid_model_dirs.append(path)
        
    return f"成功识别了{len(valid_model_dirs)}个模型：{valid_model_dirs}"

def onnx_export_func():
    model_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    output_msg = ""
    try:
        for path in model_dirs:
            pth_files = glob.glob(f"{root_dir}/{path}/*.pth")
            json_files = glob.glob(f"{root_dir}/{path}/*.json")
            model_file = Path(pth_files[0]).name
            json_file = Path(json_files[0]).name
            try:
                onnx_export(path, json_file, model_file)
                output_msg += f"成功转换{path}\n"
            except Exception as e:
                output_msg += f"转换{path}时出现错误: {e}\n"
        return output_msg
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def load_raw_audio(audio_path):
    audio_path = audio_path.replace("\"", "")
    if not os.path.isdir(audio_path):
        return "请输入正确的目录", None
    audio_files = list_files(audio_path, extensions=AUDIO_EXTENSIONS, recursive=False)
    if not audio_files:
        return "未在目录中找到音频文件", None
    
    return f"成功加载{len(audio_files)}条音频", str(os.path.join(audio_path, "output"))

def auto_slice(input_dir, output_dir, max_sec, min_sec, min_silence_duration, max_silence_kept, progress=gr.Progress()):
    if output_dir == "":
        return "请先选择输出的文件夹"
    if output_dir == input_dir:
        return "输出目录不能和输入目录相同"
    #去除路径中的引号
    output_dir = output_dir.replace("\"", "")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    audio_files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=False)
    for file in progress.tqdm(audio_files, desc="Slicing"):
        slice_audio_file_v2(
            input_file=file,
            output_dir=output_dir,
            min_duration=min_sec,
            max_duration=max_sec,
            min_silence_duration=min_silence_duration,
            top_db=-40,
            hop_length=10,
            max_silence_kept=max_silence_kept,
        )
    
    len_files, total_duration, avg_duration, min_duration, max_duration, short_files = length(output_dir, short_threshold=min_sec)
    
    if short_files:
        for i in short_files:
            os.remove(str(os.path.join(output_dir, i[3])))
            len_files -= 1
        _, _, avg_duration, min_duration, _, _ = length(output_dir)

    original_duration = 0
    for file in audio_files:
        original_duration += librosa.get_duration(filename=os.path.join(input_dir, file))

    ratio = total_duration / original_duration

    return (f"成功将音频切分为{len_files}条片段，其中最长{max_duration:.2f}秒，最短{min_duration:.2f}秒，切片后的音频总时长{total_duration/3600:.2f}小时，平均每条音频时长{avg_duration:.2f}秒\n" + 
            f"为原始音频时长的{ratio*100:.2f}%")


def model_compression(_model, is_fp16):
    if _model == "":
        return "请先选择要压缩的模型"
    else:
        model_path = os.path.join(ckpt_read_dir, _model)
        filename, extension = os.path.splitext(_model)
        output_model_name = f"{filename}_compressed{extension}"
        output_path = os.path.join(ckpt_read_dir, output_model_name)
        removeOptimizer("configs/config.json", model_path, is_fp16, output_path)
        return f"模型已成功被保存在了{output_path}"
    
def pack_autoload(model_to_pack):
    _, config_name, _ = auto_load(model_to_pack)
    if config_name == "no_config":
        return "未找到对应的配置文件，请手动选择", None
    else:
        _config = Config(os.path.join(config_read_dir, config_name), "json")
        _content = _config.read()
        spk_dict = _content["spk"]
        spk_list = ",".join(spk_dict.keys())
        return config_name, spk_list
    
def release_packing(model_to_pack, model_config, speaker, diff_to_pack, cluster_to_pack):
    model_path = diff_path = cluster_path = ""
    basename = os.path.splitext(model_to_pack)[0]
    diff_basename = os.path.splitext(diff_to_pack)[0]
    if model_to_pack == "" or model_config == "" or speaker == "":
        return "存在必选项为空，请检查后重试"
    released_pack = ReleasePacker(speaker, model_to_pack)
    released_pack.remove_temp("release_packs")
    model_path = os.path.join(ckpt_read_dir, model_to_pack)
    config_path = os.path.join(config_read_dir, model_config)
    if os.stat(model_path).st_size > 300000000:
        removeOptimizer(config_path, model_path, False, os.path.join("release_packs", model_to_pack))
        model_path = os.path.join("release_packs", model_to_pack)
    if diff_to_pack != "no_diff":
        diff_path = os.path.join(diff_read_dir, diff_to_pack)
    if cluster_to_pack != "no_cluster":
        cluster_path = os.path.join(ckpt_read_dir, cluster_to_pack)
    shutil.copyfile("configs_template/config_template.json", "release_packs/config_template.json")
    shutil.copyfile("configs_template/diffusion_template.yaml", "release_packs/diffusion_template.yaml")
    files_to_pack = [
        (model_path, f"models/{model_to_pack}"),
        (diff_path, f"models/diffusion/{diff_to_pack}") if diff_to_pack != "no_diff" else ("", ""),
        (cluster_path, f"models/{cluster_to_pack}") if cluster_to_pack != "no_cluster" else ("", ""),
        (f"release_packs/{basename}.json", f"models/{basename}.json"),
        (f"release_packs/{diff_basename}.yaml", f"models/{diff_basename}.yaml") if diff_to_pack != "no_diff" else ("", ""),
        ("release_packs/install.txt", "install.txt")
    ]
    released_pack.add_file(files_to_pack)
    released_pack.generate_config(diff_to_pack, model_config)
    os.rename("release_packs/config_template.json", f"release_packs/{basename}.json")
    os.rename("release_packs/diffusion_template.yaml", f"release_packs/{diff_basename}.yaml")
    released_pack.pack()
    to_remove = [file for file in os.listdir("release_packs") if not file.endswith(".zip")]
    for file in to_remove:
        os.remove(os.path.join("release_packs", file))
    return "打包成功, 请在release_packs目录下查看"

def release_install(model_zip_path):
    model_zip = ReleasePacker("", "")
    model_zip.unpack(model_zip_path)
    for file in os.listdir("release_packs"):
        if file.endswith(".txt"):
            install_txt = os.path.join("release_packs", file)
            break
    else:
        model_zip.remove_temp("release_packs")
        return "非格式化安装包，无法安装"
    _spk = model_zip.formatted_install(install_txt)
    model_zip.remove_temp("release_packs")
    return f"安装成功，可用说话人{_spk}，请启用独立目录模式加载模型"

def sami_inference(ac_key, s_key, app_key, audio_path, model, use_proxy, port):
    if ac_key == "" or s_key == "" or app_key == "":
        return None, "密钥和APP_KEY不能为空"
    
    if use_proxy:
        os.environ['HTTP_PROXY'] = f"http://127.0.0.1:{int(port)}/"
    
    sami_service = SAMIService()

    sami_service.set_ak(ac_key)
    sami_service.set_sk(s_key)

    auth_req = {"appkey": app_key, "token_version": 'volc-auth-v1', "expiration": 3600}
    auth_resp = sami_service.common_json_handler("GetToken", auth_req)

    try:
        auth_token = auth_resp["token"]
    except KeyError as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)
    
    payload = json.dumps({"model": model})
    with open(audio_path, "rb") as f:
        data = f.read()
        data = base64.b64encode(data).decode('utf-8')

    req = {
        "appkey": app_key,
        "token": auth_token,
        "namespace": "MusicSourceSeparate",
        "payload": payload,
        "data": data
    }

    resp = requests.post("https://sami.bytedance.com/api/v1/invoke", json=req)

    try:
        sami_resp = resp.json()
        if resp.status_code != 200:
            print(sami_resp)
            sys.exit(1)
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)
    
    print("response task_id=%s status_code=%d status_text=%s" % (
        sami_resp["task_id"], sami_resp["status_code"], sami_resp["status_text"]), end=" ")
    if "payload" in sami_resp and len(sami_resp["payload"]) > 0:
        print("payload=%s" % sami_resp["payload"], end=" ")
    if "data" in sami_resp and len(sami_resp["data"]) > 0:
        # Save audio data into file
        data = base64.b64decode(sami_resp["data"])
        print("data=[%d]bytes" % len(data))
        with open("output.wav", "wb") as f:
            f.write(data)

    if use_proxy:
        os.environ.pop('HTTP_PROXY')

    if os.path.isfile("output.wav"):
        return "output.wav", "Success"
    else:
        return None, "出错了"
    
def sami_save_fn(sami_access, sami_secret, sami_appkey):
    config_file = Config(default_settings_file, "yaml")
    default_settings = config_file.read()
    default_settings["sami_settings"]["access_key"] = sami_access
    default_settings["sami_settings"]["secret_key"] = sami_secret
    default_settings["sami_settings"]["appkey"] = sami_appkey
    config_file.save(default_settings)
    return "保存成功"
    

#read default params
sovits_params, diff_params, second_dir_enable, sami_settings = get_default_settings()
ckpt_read_dir = second_dir if second_dir_enable else workdir
config_read_dir = second_dir if second_dir_enable else config_dir
diff_read_dir = diff_second_dir if second_dir_enable else diff_workdir
current_mode = get_current_mode()

# create dirs if they don't exist
dirs_to_check = [
    workdir,
    second_dir,
    diff_workdir,
    diff_second_dir,
    dataset_dir,
]
for dir in dirs_to_check:
    if not os.path.exists(dir):
        os.makedirs(dir)

# read ckpt list
ckpt_list, config_list, cluster_list, diff_list, diff_config_list = load_options()

# read available encoder list
encoder_list = get_available_encoder()

#read GPU info
ngpu=torch.cuda.device_count()
gpu_infos=[]
if(torch.cuda.is_available() is False or ngpu==0):
    if_gpu_ok=False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name=torch.cuda.get_device_name(i)
        if("MX"in gpu_name):
            continue
        if("RTX" in gpu_name.upper() or "10"in gpu_name or "16"in gpu_name or "20"in gpu_name or "30"in gpu_name or "40"in gpu_name or "A50"in gpu_name.upper() or "70"in gpu_name or "80"in gpu_name or "90"in gpu_name or "M4"in gpu_name or"P4"in gpu_name or "T4"in gpu_name or "TITAN"in gpu_name.upper()):#A10#A100#V100#A40#P40#M40#K80
            if_gpu_ok=True#至少有一张能用的N卡
            gpu_infos.append("%s\t%s"%(i,gpu_name))
gpu_info="\n".join(gpu_infos)if if_gpu_ok is True and len(gpu_infos)>0 else "很遗憾您这没有能用的显卡来支持您训练"
gpus="-".join([i[0]for i in gpu_infos])

#read cuda info for inference
cuda = {}
min_vram = 0
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        current_vram = torch.cuda.get_device_properties(i).total_memory
        min_vram = current_vram if current_vram > min_vram else min_vram
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"
total_vram = round(min_vram * 9.31322575e-10) if min_vram != 0 else 0
auto_batch = total_vram - 2 if total_vram <= 12 and total_vram > 0 else total_vram
print(f"Current vram: {total_vram} GiB, recommended batch size: {auto_batch}")

#Check BF16 support
amp_options = ["fp32", "fp16"]
if if_gpu_ok:
    if torch.cuda.is_bf16_supported():
        amp_options = ["fp32", "fp16", "bf16"] 

#Get F0 Options
f0_options = ["crepe","pm","dio","harvest","rmvpe","fcpe"]

#Get Vocoder Options
vocoder_options = []
for dir in os.listdir("pretrain"):
    if os.path.isdir(os.path.join("pretrain", dir)):
        if os.path.isfile(os.path.join("pretrain", dir, "model")) and os.path.isfile(os.path.join("pretrain", dir, "config.json")):
            vocoder_options.append(dir)


app = gr.Blocks()
with app:
    gr.Markdown(value="""
        ### So-VITS-SVC 4.1-Stable WebUI 推理&训练 v2.3.18
                
        制作协力：bilibili@麦哲云

        仅供个人娱乐和非商业用途，禁止用于血腥、暴力、性相关、政治相关内容

        [使用文档和常见报错解答](https://www.yuque.com/umoubuton/ueupp5)

        整合包作者：bilibili@羽毛布団 | 技术交流群：742817595 | 交流二群：168254971 | 交流三群：416656175 | 交流四群：903516607

        """)
    with gr.Tabs():
        with gr.TabItem("推理") as inference_tab:
            mode_caption = gr.Markdown(value=f"""
                {current_mode}，可在页面底端切换模式
            """)
            with gr.Row():
                choice_ckpt = gr.Dropdown(label="模型选择", choices=ckpt_list, value="no_model")
                model_branch = gr.Textbox(label="模型编码器", placeholder="请先选择模型", interactive=False)
            with gr.Row():
                config_choice = gr.Dropdown(label="配置文件", choices=config_list, value="no_config")
                config_info = gr.Textbox(label="配置文件编码器", placeholder="请选择配置文件")
            gr.Markdown(value="""**请检查模型和配置文件的编码器是否匹配**""")
            with gr.Row():
                diff_choice = gr.Dropdown(label="（可选）选择扩散模型", choices=diff_list, value="no_diff", interactive=True)
                diff_config_choice = gr.Dropdown(label="扩散模型配置文件", choices=diff_config_list, value="no_diff_config", interactive=True)
            with gr.Row():
                cluster_choice = gr.Dropdown(label="（可选）选择聚类模型/特征检索模型", choices=cluster_list, value="no_clu")
                vocoder_choice = gr.Dropdown(label="选择声码器", choices=vocoder_options, value="nsf_hifigan_finetuned")
            refresh = gr.Button("刷新选项")
            refresh2 = gr.Button("刷新选项2", visible=False)
            with gr.Row():
                enhance = gr.Checkbox(label="是否使用NSF_HIFIGAN增强，该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭", value=False)
                only_diffusion = gr.Checkbox(label="是否使用全扩散推理，开启后将不使用So-VITS模型，仅使用扩散模型进行完整扩散推理，不建议使用", value=False)
            with gr.Row():
                diffusion_method = gr.Dropdown(label="扩散模型采样器", choices=["dpm-solver++","dpm-solver","pndm","ddim","unipc"], value="dpm-solver++")
                diffusion_speedup = gr.Number(label="扩散加速倍数，默认为10倍", value=10)
            using_device = gr.Dropdown(label="推理设备，默认为自动选择", choices=["Auto",*cuda.keys(),"cpu"], value="Auto")
            with gr.Row():
                loadckpt = gr.Button("加载模型", variant="primary")
                unload = gr.Button("卸载模型", variant="primary")
            with gr.Row():
                model_message = gr.Textbox(label="Output Message")
                sid = gr.Dropdown(label="So-VITS说话人", value="speaker0")
            
            inference_tab.select(refresh_options,[],[choice_ckpt,config_choice,cluster_choice,diff_choice,diff_config_choice])
            choice_ckpt.change(auto_load, [choice_ckpt], [model_branch, config_choice, config_info])  
            config_choice.change(load_json_encoder, [config_choice, choice_ckpt], [config_info])
            diff_choice.change(auto_load_diff, [diff_choice], [diff_config_choice])
            refresh.click(refresh_options,[],[choice_ckpt,config_choice,cluster_choice,diff_choice,diff_config_choice,mode_caption])
            refresh2.click(refresh_options2,[],[choice_ckpt,config_choice,cluster_choice,diff_choice,diff_config_choice,mode_caption], api_name="refresh_options")

            gr.Markdown(value="""
                请稍等片刻，模型加载大约需要10秒。后续操作不需要重新加载模型
                """)
            with gr.Tabs():
                with gr.TabItem("单个音频上传"):
                    vc_input3 = gr.Audio(label="单个音频上传", type="filepath", source="upload")
                    use_microphone = gr.Checkbox(label="使用麦克风输入")
                with gr.TabItem("批量音频上传"):
                    vc_batch_files = gr.Files(label="批量音频上传", file_types=["audio"], file_count="multiple")
                with gr.TabItem("文字转语音"):
                    gr.Markdown("""
                        文字转语音（TTS）说明：使用edge_tts服务生成音频，并转换为So-VITS模型音色。
                    """)
                    text_input = gr.Textbox(label = "在此输入需要转译的文字（建议打开自动f0预测）",)
                    with gr.Row():
                        tts_gender = gr.Radio(label = "说话人性别", choices = ["男","女"], value = "男")
                        tts_lang = gr.Dropdown(label = "选择语言，Auto为根据输入文字自动识别", choices=SUPPORTED_LANGUAGES, value = "Auto")
                    with gr.Row():
                        tts_rate = gr.Slider(label = "TTS语音变速（倍速相对值）", minimum = -1, maximum = 3, value = 0, step = 0.1)
                        tts_volume = gr.Slider(label = "TTS语音音量（相对值）", minimum = -1, maximum = 1.5, value = 0, step = 0.1)

            with gr.Row():
                auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会跑调）", value=False)
                f0_predictor = gr.Radio(label="f0预测器选择（如遇哑音可以更换f0预测器解决，crepe为原F0使用均值滤波器）", choices=f0_options, value="rmvpe")
                cr_threshold = gr.Number(label="F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音", value=0.05)
            with gr.Row():
                vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                cluster_ratio = gr.Number(label="聚类模型/特征检索混合比例，0-1之间，默认为0不启用聚类或特征检索，能提升音色相似度，但会导致咬字下降", value=0)
                k_step = gr.Slider(label="浅扩散步数，只有使用了扩散模型才有效，步数越大越接近扩散模型的结果", value=100, minimum = 1, maximum = 1000)
            with gr.Row():
                output_format = gr.Radio(label="音频输出格式", choices=["wav", "flac", "mp3"], value = "wav")
                enhancer_adaptive_key = gr.Number(label="使NSF-HIFIGAN增强器适应更高的音域(单位为半音数)|默认为0", value=0)
                slice_db = gr.Number(label="切片阈值", value=-50)
                cl_num = gr.Number(label="音频自动切片，0为按默认方式切片，单位为秒/s，爆显存可以设置此处强制切片", value=0)
            with gr.Accordion("高级设置（一般不需要动）", open=False):
                noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
                pad_seconds = gr.Number(label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5)
                lg_num = gr.Number(label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s", value=1)
                lgr_num = gr.Number(label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭", value=0.75)
                second_encoding = gr.Checkbox(label = "二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，效果时好时差，默认关闭", value=False)
                loudness_envelope_adjustment = gr.Number(label="输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络", value = 0)
                use_spk_mix = gr.Checkbox(label="动态声线融合，需要手动编辑角色混合轨道，没做完暂时不要开启", value=False, interactive=False)
            with gr.Row():
                vc_submit = gr.Button("音频转换", variant="primary")
                vc_batch_submit = gr.Button("批量转换", variant="primary")
                vc_tts_submit = gr.Button("文本转语音", variant="primary")
            #interrupt_button = gr.Button("中止转换", variant="danger")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")
        
        loadckpt.click(load_model_func,[choice_ckpt,cluster_choice,config_choice,enhance,diff_choice,diff_config_choice,only_diffusion,use_spk_mix,using_device,diffusion_method,diffusion_speedup,cl_num,vocoder_choice],[model_message, sid, cl_num, k_step], api_name="load_model")
        unload.click(model_empty_cache, [], [sid, model_message], api_name="unload_model")
        use_microphone.change(source_change, [use_microphone], [vc_input3])
        vc_submit.click(vc_fn, [output_format, sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2], api_name="run_inference")
        
        vc_batch_submit.click(vc_batch_fn, [output_format, sid, vc_batch_files, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1])
        vc_tts_submit.click(tts_fn, [text_input, tts_gender, tts_lang, tts_rate, tts_volume, output_format, sid, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2])
        #interrupt_button.click(fn=None, inputs=None, outputs=None, cancels=[vc_event])

        with gr.TabItem("训练"):
            gr.Markdown(value="""请将数据集文件夹放置在dataset_raw文件夹下，确认放置正确后点击下方获取数据集名称""")
            raw_dirs_list=gr.Textbox(label="Raw dataset directory(s):")
            get_raw_dirs=gr.Button("识别数据集", variant="primary")
            gr.Markdown(value="""确认数据集正确识别后请选择训练使用的特征编码器和f0预测器，**如果要训练扩散模型，请选择Vec768l12或hubertsoft或whisper-ppg，并确保So-VITS和扩散模型使用同一个编码器**""")
            with gr.Row():
                gr.Markdown(value="""**vec256l9**: ContentVec(256Layer9)，旧版本叫v1，So-VITS-SVC 4.0的基础版本，**不推荐使用**
                                **vec768l12**: 特征输入更换为ContentVec的第12层Transformer输出，模型理论上会更加还原训练集音色
                                **hubertsoft**: So-VITS-SVC 3.0使用的编码器，咬字更为准确，但可能存在多说话人音色泄露问题
                                **whisper-ppg**: 来自OpenAI，咬字最为准确，但和Hubertsoft一样存在多说话人音色泄露，且显存占用和训练时间有明显增加。
                                解锁更多编码器选项，请见[这里](https://www.yuque.com/umoubuton/ueupp5/kmui02dszo5zrqkz)
                """)
                gr.Markdown(value="""**crepe**: 抗噪能力最强，但预处理速度慢（不过如果你的显卡很强的话速度会很快）
                                **pm**: 预处理速度快，但抗噪能力较弱
                                **dio**: 先前版本预处理默认使用的f0预测器，比较拉胯不推荐使用
                                **harvest**: 有一定抗噪能力，预处理显存占用友好，速度比较慢
                                **rmvpe**: 最精准的预测器，crepe的完全上位替代
                                **fcpe**: SVC开发组自研F0预测器，有最快的速度和不输crepe的精度
                """)
            with gr.Row():
                branch_selection = gr.Dropdown(label="选择训练使用的编码器", choices=encoder_list, value="vec768l12", interactive=True)
                f0_predictor_selection = gr.Dropdown(label="选择训练使用的f0预测器", choices=f0_options, value="rmvpe", interactive=True)
            with gr.Row():
                use_diff = gr.Checkbox(label="是否使用浅扩散模型，如要训练浅扩散请勾选此项，将会在预处理时生成浅扩散必备的特征文件（确定不训练可以不勾，能节省一点空间）", value=True)
                vol_aug=gr.Checkbox(label="是否启用响度嵌入和音量增强，启用后可以根据输入源控制输出响度，但对数据集质量的要求更高。**仅支持vec768l12编码器**", value=False)
                tiny_enable = gr.Checkbox(label="是否启用TINY训练，TINY为实时专用模型，显存占用更低，推理速度更快，但质量有所削减。仅支持vec768，且必须打开响度嵌入", value=False)
            with gr.Row():
                skip_loudnorm = gr.Checkbox(label="是否跳过响度匹配，如果你已经用音频处理软件做过响度匹配，请勾选此处")
                num_processes = gr.Slider(label="预处理使用的CPU线程数，可以大幅加快预处理速度，但线程数过大容易爆显存，建议12G显存设置为2", minimum=1, maximum=multiprocessing.cpu_count(), value=1, step=1)
            with gr.Row():
                raw_preprocess=gr.Button("数据预处理", variant="primary")
                regenerate_config_btn=gr.Button("重新生成配置文件", variant="primary")
            preprocess_output=gr.Textbox(label="预处理输出信息，完成后请检查一下是否有报错信息，如无则可以进行下一步", max_lines=999)
            clear_preprocess_output=gr.Button("清空输出信息")
            with gr.Group():
                gr.Markdown(value="""填写训练设置和超参数""")
                with gr.Row():
                    gr.Textbox(label="当前使用显卡信息", value=gpu_info)
                    gpu_selection=gr.Textbox(label="多卡用户请指定希望训练使用的显卡ID（0,1,2...）", value=gpus, interactive=True)
                with gr.Row():
                    log_interval=gr.Textbox(label="每隔多少步(steps)生成一次评估日志", value=sovits_params['log_interval'])
                    eval_interval=gr.Textbox(label="每隔多少步(steps)验证并保存一次模型", value=sovits_params['eval_interval'])
                    keep_ckpts=gr.Textbox(label="仅保留最新的X个模型，超出该数字的旧模型会被删除。设置为0则永不删除", value=sovits_params['keep_ckpts'])
                with gr.Row():
                    batch_size=gr.Textbox(label="批量大小，每步取多少条数据进行训练，大batch有助于训练但显著增加显存占用。6G显存建议设定为4", value=auto_batch)
                    lr=gr.Textbox(label="学习率，一般不用动，批量大小较大时可以适当增大学习率，但强烈不建议超过0.0002，有炸炉风险", value=sovits_params['learning_rate'])
                    amp_dtype = gr.Radio(label="训练数据类型，fp16可能会有更快的训练速度和更低的显存占用，但容易炸炉", choices=amp_options, value=sovits_params['amp_dtype'])
                    all_in_mem=gr.Checkbox(label="是否加载所有数据集到内存中，硬盘IO过于低下、同时内存容量远大于数据集体积时可以启用，能显著加快训练速度", value=sovits_params['all_in_mem'])
                with gr.Row():
                    gr.Markdown("请检查右侧的说话人列表是否和你要训练的目标说话人一致，确认无误后点击写入配置文件，然后就可以开始训练了")
                    speakers=gr.Textbox(label="说话人列表")
            with gr.Accordion(label = "扩散模型配置（训练扩散模型需要写入此处）", open=True):
                with gr.Row():
                    diff_num_workers = gr.Number(label="num_workers, 如果你的电脑配置较高，可以将这里设置为0加快训练速度", value=diff_params['num_workers'])
                    diff_k_step_max = gr.Checkbox(label="只训练100步深度的浅扩散模型。能加快训练速度并提高模型质量，代价是无法执行超过100步的浅扩散推理", value=diff_params['diff_k_step_max'])
                    diff_cache_all_data = gr.Checkbox(label="是否缓存数据，启用后可以加快训练速度，关闭后可以节省显存或内存，但会减慢训练速度", value=diff_params['cache_all_data'])
                    diff_cache_device = gr.Radio(label="若启用缓存数据，使用显存(cuda)还是内存(cpu)缓存，如果显卡显存充足，选择cuda以加快训练速度", choices=["cuda","cpu"], value=diff_params['cache_device'])
                    diff_amp_dtype = gr.Radio(label="训练数据类型，fp16可能会有更快的训练速度，前提是你的显卡支持", choices=["fp32","fp16"], value=diff_params['amp_dtype'])
                with gr.Row():
                    diff_batch_size = gr.Number(label="批量大小(batch_size)，根据显卡显存设置，小显存适当降低该项，6G显存可以设定为48，但该数值不要超过数据集总数量的1/4", value=diff_params['diff_batch_size'])
                    diff_lr = gr.Number(label="学习率（一般不需要动）", value=diff_params['diff_lr'])
                    diff_interval_log = gr.Number(label="每隔多少步(steps)生成一次评估日志", value = diff_params['diff_interval_log'])
                    diff_interval_val = gr.Number(label="每隔多少步(steps)验证并保存一次模型，如果你的批量大小较大，可以适当减少这里的数字，但不建议设置为1000以下", value=diff_params['diff_interval_val'])
                    diff_force_save = gr.Number(label="每隔多少步强制保留模型，只有该步数的倍数保存的模型会被保留，其余会被删除。设置为与验证步数相同的值则每个模型都会被保留", value=diff_params['diff_force_save'])
            with gr.Row():
                save_params=gr.Button("将当前设置保存为默认设置", variant="primary")
                write_config=gr.Button("写入配置文件", variant="primary")
            write_config_output=gr.Textbox(label="输出信息")

            gr.Markdown(value="""**点击从头开始训练**将会自动将已有的训练进度保存到models_backup文件夹，并自动装载预训练模型。
                **继续上一次的训练进度**将从上一个保存模型的进度继续训练。继续训练进度无需重新预处理和写入配置文件。
                关于扩散、聚类和特征检索的详细说明请看[此处](https://www.yuque.com/umoubuton/ueupp5/kmui02dszo5zrqkz)。
                """)
            with gr.Row():
                with gr.Column():
                    start_training=gr.Button("从头开始训练", variant="primary")
                    training_output=gr.Textbox(label="训练输出信息")
                with gr.Column():
                    continue_training_btn=gr.Button("继续上一次的训练进度", variant="primary")
                    continue_training_output=gr.Textbox(label="训练输出信息")
            with gr.Row():
                with gr.Column():
                    diff_training_btn=gr.Button("从头训练扩散模型", variant="primary")
                    diff_training_output=gr.Textbox(label="训练输出信息")
                with gr.Column():
                    diff_continue_training_btn=gr.Button("继续训练扩散模型", variant="primary")
                    diff_continue_training_output=gr.Textbox(label="训练输出信息") 
            with gr.Accordion(label = "聚类、特征检索训练", open=False):
                with gr.Row():               
                    with gr.Column():
                        kmeans_button=gr.Button("训练聚类模型", variant="primary")
                        kmeans_gpu = gr.Checkbox(label="使用GPU训练", value=True)
                        kmeans_output=gr.Textbox(label="训练输出信息")
                    with gr.Column():
                        index_button=gr.Button("训练特征检索模型", variant="primary")
                        index_output=gr.Textbox(label="训练输出信息")

        with gr.TabItem("小工具/实验室特性"):
            gr.Markdown(value="""
                        ### So-vits-svc 4.1 小工具/实验室特性
                        提供了一些有趣或实用的小工具，可以自行探索
                        """)
            with gr.Tabs():
                with gr.TabItem("静态声线融合"):
                    gr.Markdown(value="""
                        <font size=2> 介绍:该功能可以将多个声音模型合成为一个声音模型(多个模型参数的凸组合或线性组合)，从而制造出现实中不存在的声线 
                                          注意：
                                          1.该功能仅支持单说话人的模型
                                          2.如果强行使用多说话人模型，需要保证多个模型的说话人数量相同，这样可以混合同一个SpaekerID下的声音
                                          3.保证所有待混合模型的config.json中的model字段是相同的
                                          4.输出的混合模型可以使用待合成模型的任意一个config.json，但聚类模型将不能使用
                                          5.批量上传模型的时候最好把模型放到一个文件夹选中后一起上传
                                          6.混合比例调整建议大小在0-100之间，也可以调为其他数字，但在线性组合模式下会出现未知的效果
                                          7.混合完毕后，文件将会保存在项目根目录中，文件名为output.pth
                                          8.凸组合模式会将混合比例执行Softmax使混合比例相加为1，而线性组合模式不会
                        </font>
                        """)
                    mix_model_path = gr.Files(label="选择需要混合模型文件")
                    mix_model_upload_button = gr.UploadButton("选择/追加需要混合模型文件", file_count="multiple")
                    mix_model_output1 = gr.Textbox(
                                            label="混合比例调整，单位/%",
                                            interactive = True
                                         )
                    mix_mode = gr.Radio(choices=["凸组合", "线性组合"], label="融合模式",value="凸组合",interactive = True)
                    mix_submit = gr.Button("声线融合启动", variant="primary")
                    mix_model_output2 = gr.Textbox(
                                            label="Output Message"
                                         )
                with gr.TabItem("onnx转换"):
                    gr.Markdown(value="""
                        提供了将.pth模型（批量）转换为.onnx模型的功能
                        源项目本身自带转换的功能，但不支持批量，操作也不够简单，这个工具可以支持在WebUI中以可视化的操作方式批量转换.onnx模型
                        有人可能会问，转.onnx模型有什么作用呢？相信我，如果你问出了这个问题，说明这个工具你应该用不上

                        ### Step 1: 
                        在整合包根目录下新建一个"checkpoints"文件夹，将pth模型和对应的json配置文件按目录分别放置到checkpoints文件夹下
                        看起来应该像这样：
                        checkpoints
                        ├───xxxx
                        │   ├───xxxx.pth
                        │   └───xxxx.json
                        ├───xxxx
                        │   ├───xxxx.pth
                        │   └───xxxx.json
                        └───……
                        """)
                    pth_dir_msg = gr.Textbox(label="识别待转换模型", placeholder="请将模型和配置文件按上述说明放置在正确位置")
                    pth_dir_identify_btn = gr.Button("识别", variant="primary")
                    gr.Markdown(value="""
                        ### Step 2:
                        识别正确后点击下方开始转换，转换一个模型可能需要一分钟甚至更久
                        """)
                    pth2onnx_btn = gr.Button("开始转换", variant="primary")
                    pth2onnx_msg = gr.Textbox(label="输出信息")

                with gr.TabItem("智能音频切片"):
                    gr.Markdown(value="""
                        该工具可以实现对音频的切片，无需调整参数即可完成符合要求的数据集制作。整合自冷月佬的[Fish Audio Preprocessor](https://github.com/fishaudio/audio-preprocess)
                        数据集要求的音频切片约在2-15秒内，用传统的Slicer-GUI切片工具需要精准调参和二次切片才能符合要求，该工具省去了上述繁琐的操作，只要上传原始音频即可一键制作数据集。
                    """)
                    with gr.Row():
                        raw_audio_path = gr.Textbox(label="原始音频文件夹", placeholder="包含所有待切片音频的文件夹，示例: D:\干声\speakers")
                        load_raw_audio_btn = gr.Button("加载原始音频", variant = "primary")
                    load_raw_audio_output = gr.Textbox(label = "输出信息")
                    #raw_audio_dataset = gr.Textbox(label = "音频数据集", value=None, interactive=False)
                    slicer_output_dir = gr.Textbox(label = "输出目录", placeholder = "选择输出目录（不要和输入音频是同一个文件夹）")
                    with gr.Row():
                        max_sec = gr.Number(label = "切片的最长秒数", value = 15)
                        min_sec = gr.Number(label = "切片的最短秒数", value = 2)
                    with gr.Accordion(label="高级设置", open=False):
                        min_silence_dur = gr.Number(label = "视为静音的最短长度（秒）", value = 0.3)
                        max_silence_kept = gr.Number(label = "切片音频周围保持的最大静音长度（秒）", value = 1.0)
                    slicer_btn = gr.Button("开始切片", variant = "primary")
                    slicer_output_msg = gr.Textbox(label = "输出信息")

                    mix_model_path.change(updata_mix_info,[mix_model_path],[mix_model_output1])
                    mix_model_upload_button.upload(upload_mix_append_file, [mix_model_upload_button,mix_model_path], [mix_model_path,mix_model_output1])
                    mix_submit.click(mix_submit_click, [mix_model_output1,mix_mode], [mix_model_output2])
                    pth_dir_identify_btn.click(pth_identify, [], [pth_dir_msg])
                    pth2onnx_btn.click(onnx_export_func, [], [pth2onnx_msg])
                    load_raw_audio_btn.click(load_raw_audio, [raw_audio_path], [load_raw_audio_output, slicer_output_dir])
                    slicer_btn.click(auto_slice, [raw_audio_path, slicer_output_dir, max_sec, min_sec, min_silence_dur, max_silence_kept], [slicer_output_msg])
                
                with gr.TabItem("模型压缩工具"):
                    gr.Markdown(value="""
                        该工具可以实现对模型的体积压缩，在**不影响模型推理功能**的情况下，将原本约600M的So-VITS模型压缩至约200M, 大大减少了硬盘的压力。
                        **注意：压缩后的模型将无法继续训练，请在确认封炉后再压缩。**
                        将模型文件放置在logs/44k下，然后选择需要压缩的模型
                    """)
                    model_to_compress = gr.Dropdown(label="模型选择", choices=ckpt_list, value="no_model")
                    fp16_compress = gr.Checkbox(label="使用 fp16 压缩", value=False)
                    compress_model_btn = gr.Button("压缩模型", variant="primary")
                    compress_model_output = gr.Textbox(label="输出信息", value="")

                    compress_model_btn.click(model_compression, [model_to_compress, fp16_compress], [compress_model_output])

                with gr.TabItem("模型发布打包/安装"):
                    gr.Markdown(value="""
                        如果你想将你的模型分享给他人，请使用该工具对模型进行打包。
                        该工具可以自动生成正确的配置文件，确保你在打包过程中不出现任何遗漏和错误，接收到使用该工具打包的模型后，也可以用该工具进行自动安装。
                    """)
                    with gr.Tabs():
                        with gr.TabItem("安装"):
                            with gr.Row():
                                model_to_install = gr.Textbox(label = "模型压缩包路径", placeholder="示例：D:\Downloads\model_packing.zip") 
                                install_model_btn = gr.Button("安装", variant="primary")
                            install_output = gr.Textbox(label="输出信息", value="")
                        with gr.TabItem("打包"):
                            with gr.Row():
                                model_to_pack = gr.Dropdown(label="选择要打包的模型", choices=ckpt_list, value="")
                                model_config = gr.Dropdown(label="选择要打包的模型配置文件", choices=config_list, value="", interactive=True)
                                speaker_name = gr.Textbox(label="模型说话人名称", placeholder="该模型的说话人名称，仅限数字字母下划线，如模型中有多说话人，请用逗号分割，例如：spk1,spk2,spk3", value = "")
                            with gr.Row():
                                diff_to_pack = gr.Dropdown(label="（可选）选择要打包的扩散模型", choices=diff_list, value="no_diff")
                                cluster_to_pack = gr.Dropdown(label="（可选）选择要打包的聚类或特征检索模型", choices=cluster_list, value="no_cluster")
                            packing_btn = gr.Button("开始打包", variant="primary")
                            packing_output_msg = gr.Textbox(label = "输出信息")
                   
                    model_to_pack.change(pack_autoload, [model_to_pack], [model_config, speaker_name])
                    packing_btn.click(release_packing, [model_to_pack, model_config, speaker_name, diff_to_pack, cluster_to_pack], [packing_output_msg])
                    install_model_btn.click(release_install, [model_to_install], [install_output])

                with gr.TabItem("歌曲人声分离"):
                    gr.Markdown(value="""
                        使用火山引擎 SAMI 技术分离人声，需要联网并自行创建应用 API 后使用。
                        详细使用文档请见[这里](https://www.yuque.com/umoubuton/ueupp5/zmxisze10hlfidsk)
                    """)
                    with gr.Row():
                        input_audio = gr.Audio(label="上传原始音频", type="filepath", source="upload")
                        sami_model = gr.Dropdown(label="选择分离模型", choices=["2track_vocal","2track_acc","bs_4track_vocal","bs_4track_acc"], value="bs_4track_vocal")
                    with gr.Row():
                        sami_access = gr.Textbox(label="Access Key", value=sami_settings["access_key"])
                        sami_secret = gr.Textbox(label="Secret Key", value=sami_settings["secret_key"])
                        sami_appkey = gr.Textbox(label="App Key", value=sami_settings["appkey"])
                    with gr.Row():
                        use_proxy = gr.Checkbox(label="使用代理", value=False)
                        proxy_port = gr.Number(label="代理端口", value=7890)

                    sami_submit = gr.Button("开始分离", variant="primary")
                    sami_save_config = gr.Button("保存密钥为默认配置")
                    sami_output = gr.Audio(label="输出结果", type="filepath")
                    sami_output_msg = gr.Textbox(label="输出信息")

                    sami_submit.click(sami_inference, [sami_access, sami_secret, sami_appkey, input_audio, sami_model, use_proxy, proxy_port], [sami_output, sami_output_msg])
                    sami_save_config.click(sami_save_fn, [sami_access, sami_secret, sami_appkey], [sami_output_msg])


        get_raw_dirs.click(load_raw_dirs,[],[raw_dirs_list])
        raw_preprocess.click(dataset_preprocess,[branch_selection, f0_predictor_selection, use_diff, vol_aug, skip_loudnorm, num_processes, tiny_enable],[preprocess_output, speakers])
        regenerate_config_btn.click(regenerate_config,[branch_selection, vol_aug, tiny_enable],[preprocess_output])
        clear_preprocess_output.click(clear_output,[],[preprocess_output])
        save_params.click(save_default_settings, [log_interval,eval_interval,keep_ckpts,batch_size,lr,amp_dtype,all_in_mem,diff_num_workers,diff_cache_all_data,diff_cache_device,diff_amp_dtype,diff_batch_size,diff_lr,diff_interval_log,diff_interval_val,diff_force_save,diff_k_step_max], [write_config_output])
        write_config.click(config_fn,[log_interval, eval_interval, keep_ckpts, batch_size, lr, amp_dtype, all_in_mem, diff_num_workers, diff_cache_all_data, diff_batch_size, diff_lr, diff_interval_log, diff_interval_val, diff_cache_device, diff_amp_dtype, diff_force_save, diff_k_step_max],[write_config_output])
        start_training.click(training,[gpu_selection, branch_selection, tiny_enable],[training_output])
        diff_training_btn.click(diff_training,[branch_selection, diff_k_step_max],[diff_training_output])
        continue_training_btn.click(continue_training,[gpu_selection, branch_selection],[continue_training_output])
        diff_continue_training_btn.click(diff_continue_training,[branch_selection],[diff_continue_training_output])
        kmeans_button.click(kmeans_training,[kmeans_gpu],[kmeans_output])
        index_button.click(index_training, [], [index_output])

    with gr.Tabs():
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown(value="""
                    <font size=2> WebUI设置</font>
                    """)
                with gr.Row():
                    debug_button = gr.Checkbox(label="Debug模式，反馈BUG需要打开，打开后控制台可以显示具体错误提示", value=debug)
                    read_second_dir = gr.Checkbox(label = "独立目录模式，开启后将从独立目录（./models）读取模型和配置文件，变更后需要刷新选项才能生效", value=second_dir_enable)
        debug_button.change(debug_change,[],[])
        read_second_dir.change(webui_change,[read_second_dir],[])

        app.queue(concurrency_count=1022, max_size=2044).launch(server_name="127.0.0.1",server_port=7865,inbrowser=True,quiet=True)