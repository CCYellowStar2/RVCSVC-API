import re, os, time
import requests
import json
import torch
import argparse
from difflib import SequenceMatcher

parser = argparse.ArgumentParser()
parser.add_argument(
    '--is_nohalf', action='store_true'
)
a = parser.parse_args()
is_half=not a.is_nohalf
device = 'cuda' if torch.cuda.is_available() else 'cpu'
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}
pattern = r'//www\.bilibili\.com/video[^"]*'
models=[]
index=[]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RVC_API_BASE = "http://127.0.0.1:7897"

def optimize_pitch_shift(key_shift):
    if key_shift > 6:
        return key_shift - 12
    elif key_shift < -6:
        return key_shift + 12
    else:
        return key_shift

def get_response(song_id):
  print("开始下载歌曲")
  try:
    response = requests.get(f"https://biliplayer.91vrchat.com/player/?url=https://music.163.com/song?id={song_id}",allow_redirects=True, timeout=30)
    if response.status_code == 200:
      return response
  except Exception as e:
    print(f"主源下载失败: {e}")
  
  print("使用备用源下载歌曲")
  try:
      response1 = requests.get(
          f"https://api.vkeys.cn/v2/music/netease?id={song_id}",
          timeout=30
      ).json()["data"]["url"]
      res = requests.get(response1, timeout=30)
      return res
  except Exception as e:
      raise Exception(f"所有下载源均失败: {e}")

def cookie_to_dict(cookie_string):
    cookie_dict = {}
    for item in cookie_string.split(';'):
        try:
            key, value = item.strip().split('=', 1)
            cookie_dict[key] = value
        except ValueError:
            pass
    return cookie_dict
    
def change_model(model):
  try:
    response = requests.post(f"{RVC_API_BASE}/run/infer_change_voice", json={
      "data": [
        model,
        0.33,
        0.33,
    ]}, timeout=10).json()
    print(f"模型已切换为: {model}")
    return f"✅ 成功切换到模型: {model}"
  except Exception as e:
    print(f"切换模型失败: {e}")
    return f"❌ 切换模型失败: {e}"

def show_model():
  global models, index
  try:
    response = requests.post(f"{RVC_API_BASE}/run/infer_refresh", json={
      "data": []
    }, timeout=10).json()

    models = response["data"][0]["choices"]
    index = response["data"][1]["choices"]
    print(f"已加载 {len(models)} 个模型")
    return models
  except Exception as e:
    print(f"获取模型列表失败: {e}")
    return []

def find_index(model):   
    if not index:
        return None
    model_name = os.path.splitext(model)[0].lower()
    best_match = None
    best_score = 0
    threshold = 0.4
    for index_path in index:
        index_name = os.path.splitext(os.path.basename(index_path))[0].lower()
        score = SequenceMatcher(None, model_name, index_name).ratio()
        if score > best_score:
            best_score = score
            best_match = index_path
    if best_score < threshold:
        print(f"未找到匹配的 index（最高相似度: {best_score:.2f}）")
        return None
    if best_match:
        best_match="./"+ best_match
        print(f"找到匹配: {best_match}（相似度: {best_score:.2f}）")
    return best_match
    

from uvr5.vr import AudioPre
from msst_backend import MSST_SELECTION, separate_msst_audio
weight_uvr5_root = "uvr5/uvr_model"

def download_file_openxlab(url, destination):
  if os.path.exists(destination):
    return
  directory = os.path.dirname(destination)
  if directory:
    os.makedirs(directory, exist_ok=True)
  while True:
    try:
      response = requests.get(url, timeout=60)
      if response.status_code == 200:
        with open(destination, "wb") as file:
          file.write(response.content)
        return
      print(f"下载 VR-HP5 模型失败，HTTP {response.status_code}")
    except Exception as error:
      print(f"下载 VR-HP5 模型失败: {error}")
    time.sleep(5)

download_file_openxlab(
  "https://modelscope.cn/api/v1/models/CCYellowStar/5_HP-Karaoke-UVR/repo?Revision=master&FilePath=5_HP-Karaoke-UVR.pth",
  os.path.join(weight_uvr5_root, "5_HP-Karaoke-UVR.pth"),
)

# 初始化 VR 模型
func = AudioPre
pre_fun_hp5 = func(
  agg=int(10),
  model_path=os.path.join(weight_uvr5_root, "5_HP-Karaoke-UVR.pth"),
  device=device,
  is_half=is_half,
)

from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.effects import compress_dynamic_range
from pydub.effects import normalize
from pedalboard import Pedalboard, Compressor, Reverb
from scipy.signal import firwin, lfilter, iirfilter
import os
import numpy as np
import librosa
import soundfile
import gradio as gr
import scipy.signal
if not hasattr(scipy.signal, 'hann'):
    scipy.signal.hann = np.hanning

def wwy_downloader(filename, uvr_choice):
    if uvr_choice == MSST_SELECTION:
        split_model = "MSST-BS-RoFormer"
        print("🚀 使用 MSST BS-Roformer 模型进行分离...")
    else:
        split_model = "UVR-HP5"
        print("🚀 使用 VR-HP5 模型进行分离...")

    audio_content = get_response(filename).content
    temp_prefixed_path = "rvc_" + filename.strip() + ".wav"
    with open(temp_prefixed_path, mode="wb") as f:
        f.write(audio_content)
    
    audio_orig = AudioSegment.from_file(temp_prefixed_path)
    duration_minutes = len(audio_orig) / 60000
    print(f"原始音频时长: {duration_minutes:.2f} 分钟")
    if duration_minutes > 5:
        print("⚠️ 音频超过5分钟，正在截取前5分钟...")
        audio_orig = audio_orig[:300000]
    
    uvr_input_path = filename.strip() + ".wav"
    audio_orig.export(uvr_input_path, format="wav")
    
    if os.path.isfile(temp_prefixed_path):
        os.remove(temp_prefixed_path)

    output_dir = f"./output/{split_model}/{filename}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 执行分离 ==========
    if uvr_choice == MSST_SELECTION:
        vocal_path, inst_path = separate_msst_audio(
            uvr_input_path,
            output_dir,
            model_selection=uvr_choice,
        )
    else:
        # VR 模型分离
        pre_fun_hp5._path_audio_(uvr_input_path, output_dir, output_dir, "wav")
        vocal_path = f"{output_dir}/vocal_{filename}.wav_10.wav"
        inst_path = f"{output_dir}/instrument_{filename}.wav_10.wav"
    # ==============================
    
    if os.path.isfile(uvr_input_path):
        os.remove(uvr_input_path)

    return vocal_path, inst_path



def convert(song_name_src, key_shift, vocal_vol, inst_vol, model_dropdown, reverb_intensity, delay_intensity, uvr_choice="VR-HP5 (默认)"):
  
  # 映射选择到内部标识
  if uvr_choice == MSST_SELECTION:
      current_uvr_mode = MSST_SELECTION
      split_model_dir = "MSST-BS-RoFormer"
  else:
      current_uvr_mode = "VR-HP5"
      split_model_dir = "UVR-HP5"

  if not song_name_src: raise gr.Error("请输入歌曲ID或链接！")
  
  if song_name_src.startswith("http"):
    try: song_name_src = song_name_src.split('id=')[1].split('&')[0]
    except IndexError: raise gr.Error("无效的网易云链接格式！")
  
  song_name_src = song_name_src.strip()
  print(f"处理歌曲ID: {song_name_src} | 使用分离模型: {current_uvr_mode}")
  
  audio_rvc_path = os.path.join(SCRIPT_DIR, "audio_rvc.wav")
  
  # === 智能判断缓存路径 ===
  if current_uvr_mode == MSST_SELECTION:
      expected_vocal_path = f"./output/{split_model_dir}/{song_name_src}/{song_name_src}_vocals.wav"
      expected_inst_path = f"./output/{split_model_dir}/{song_name_src}/{song_name_src}_other.wav"
  else:
      expected_vocal_path = f"./output/{split_model_dir}/{song_name_src}/vocal_{song_name_src}.wav_10.wav"
      expected_inst_path = f"./output/{split_model_dir}/{song_name_src}/instrument_{song_name_src}.wav_10.wav"

  if os.path.isfile(expected_vocal_path) and os.path.isfile(expected_inst_path):
    print(f"✅ 歌曲已缓存 ({current_uvr_mode})，跳过下载")
    vocal_path = expected_vocal_path
    inst_path = expected_inst_path
    audio, sr = librosa.load(vocal_path, sr=44100, mono=True)
    soundfile.write(audio_rvc_path, audio, sr)
  else:
    print("📥 未找到缓存，开始下载和分离")
    vocal_path, inst_path = wwy_downloader(song_name_src, current_uvr_mode)
    audio_rvc, sr_src = librosa.load(vocal_path, sr=44100, mono=True)
    soundfile.write(audio_rvc_path, audio_rvc, sr_src)

  print("🎤 RVC 推理中...")
  switch_model(model_dropdown)
  response = requests.post(f"{RVC_API_BASE}/run/infer_convert", json={
    "data": [
      0,
      audio_rvc_path,
      key_shift,
      None,
      "rmvpe",
      "",
      find_index(model_dropdown),
      0.75,
      3,
      0,
      0.25,
      0.33,
  ]}).json()

  data = response["data"][1]["name"]

  if data:
    print("🎛️ 开始处理音频")
    os.makedirs("./temp", exist_ok=True)

    audio_data, sr = librosa.load(data, sr=None, mono=False)

    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(1, -1)

    from pedalboard import Pedalboard, Compressor, Reverb, HighpassFilter, PeakFilter, LowpassFilter, PitchShift, Delay

    room_size_map =  (0.15,          0.40,          0.90)
    wet_level_map =  (0.10,          0.25,          0.45)

    if reverb_intensity <= 4:
        percent = reverb_intensity / 4.0
        room_size_val = room_size_map[0] + (room_size_map[1] - room_size_map[0]) * percent
        wet_level_val = wet_level_map[0] + (wet_level_map[1] - wet_level_map[0]) * percent
    else:
        percent = (reverb_intensity - 4) / 6.0 
        room_size_val = room_size_map[1] + (room_size_map[2] - room_size_map[1]) * percent
        wet_level_val = wet_level_map[1] + (wet_level_map[2] - wet_level_map[1]) * percent

    dry_level_val = 1.0 - wet_level_val

    print(f"🎤 混响设置: 强度 {reverb_intensity}/10 => 房间大小={room_size_val:.2f}, 湿润度={wet_level_val:.2f}")
    
    effects = [
        HighpassFilter(cutoff_frequency_hz=80),
        PeakFilter(cutoff_frequency_hz=200, gain_db=1.5, q=0.7),
        PeakFilter(cutoff_frequency_hz=3000, gain_db=2.0, q=1.0),
        PeakFilter(cutoff_frequency_hz=7000, gain_db=-3.0, q=2.0),
        LowpassFilter(cutoff_frequency_hz=16000),
        Compressor(
            threshold_db=-18.0,
            ratio=4.0,
            attack_ms=5.0,
            release_ms=150.0
        ),
    ]
    
    if delay_intensity > 0:
        print("🎤 启用回声效果...")
        try:
            y_inst, sr_inst = librosa.load(inst_path, sr=None)
            tempo, _ = librosa.beat.beat_track(y=y_inst, sr=sr_inst)
            actual_tempo = tempo[0] if isinstance(tempo, np.ndarray) else tempo

            if actual_tempo > 0:
                delay_seconds_val = (60.0 / actual_tempo) * 0.5 
            else:
                delay_seconds_val = 0.5
        except Exception:
            delay_seconds_val = 0.5
            
        delay_mix_val = (delay_intensity / 10.0) * 0.35
        effects.append(Delay(delay_seconds=delay_seconds_val, feedback=0.25, mix=delay_mix_val))
    
    effects.append(Reverb(room_size=room_size_val, damping=0.4, wet_level=wet_level_val, dry_level=dry_level_val, width=0.8))
    
    board = Pedalboard(effects)
    processed = board(audio_data, sr)
    processed_int16 = (processed.T * 32768).astype(np.int16)
    processed_audio = AudioSegment(processed_int16.tobytes(), frame_rate=sr, sample_width=2, channels=processed.shape[0])
    
    audio_vocal_adjusted = processed_audio + vocal_vol
    normalized_audio = normalize(audio_vocal_adjusted, headroom=-1.0)
    
    print("🎵 准备伴奏...")
    key_shift = optimize_pitch_shift(key_shift)
    
    if key_shift != 0 and abs(key_shift) != 12:
        print(f"🎹 正在将伴奏音高调整 {key_shift:+d} 半音...")
        try:
            y_inst, sr_inst = librosa.load(inst_path, sr=None)
            pitch_board = Pedalboard([PitchShift(semitones=key_shift)])
            y_shifted = pitch_board(y_inst, sr_inst)
            
            shifted_inst_path = f"temp/shifted_{song_name_src}_inst.wav"
            soundfile.write(shifted_inst_path, y_shifted, sr_inst)
            audio_inst = AudioSegment.from_file(shifted_inst_path, format="wav")
        except Exception as e:
            print(f"⚠️ 伴奏调整失败: {e}")
            audio_inst = AudioSegment.from_file(inst_path, format="wav")
    else:
        audio_inst = AudioSegment.from_file(inst_path, format="wav")

    audio_inst = audio_inst + inst_vol
    combined_audio = normalized_audio.overlay(audio_inst)

    print("💾 导出最终文件...")
    output_path = f"temp/{sanitize_filename(song_name_src)}-RVC-AI翻唱.mp3"
    combined_audio.export(output_path, format="MP3", bitrate="192k")
    
    if os.path.isfile(data):
      os.remove(data)
    
    print(f"✅ 已导出: {output_path}")
    return output_path

def sanitize_filename(filename):
    return re.sub(r'[\\/:*?"<>|]', '', filename)

def refresh_models():
    models_list = show_model()
    if models_list:
        return gr.Dropdown(choices=models_list, value=models_list[0] if models_list else None)
    else:
        return gr.Dropdown(choices=["无可用模型"], value="无可用模型")

def switch_model(model_name):
    if not model_name or model_name == "无可用模型":
        return "❌ 请先选择一个有效的模型"
    return change_model(model_name)
    
app = gr.Blocks()

with app:
  gr.Markdown("# <center>RVC一键翻唱、重磅更新！</center>")
  gr.Markdown("## 自动分离人声翻唱并合并，自动混音！</center>")
  
  with gr.Row():
    with gr.Column():
      with gr.Row():
        model_dropdown = gr.Dropdown(label="选择AI模型", choices=[], value=None, info="请先点击刷新加载模型列表")
        refresh_btn = gr.Button("🔄 刷新", size="sm")
        switch_btn = gr.Button("✨ 切换模型", size="sm", variant="primary")
      with gr.Row(visible=False):
          models_json = gr.JSON()
          get_models_btn = gr.Button("获取模型列表", visible=False)
          get_models_btn.click(show_model, outputs=models_json)
      with gr.Row():
        model_status = gr.Textbox(label="模型状态", value="请选择模型", interactive=False)
      
      with gr.Row():
        inp1 = gr.Textbox(label="请填写想要AI翻唱的网易云id或链接", placeholder="114514", info="直接填写网易云id或链接")
      
      # ========== 新增：选择分离模型 ==========
      with gr.Row():
          uvr_model_choice = gr.Radio(
              label="人声分离模型 (UVR)",
              choices=["VR-HP5 (默认)", MSST_SELECTION],
              value="VR-HP5 (默认)",
              info="MSST BS-RoFormer：主唱优先，适合翻唱时保留和声"
          )
      # ======================================

      with gr.Row():
        inp5 = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="歌曲人声升降调", info="默认为0，+2为升高2个key，以此类推")
        inp6 = gr.Slider(minimum=-3, maximum=3, value=0, step=0.5, label="调节人声音量，默认为0")
        inp7 = gr.Slider(minimum=-3, maximum=3, value=0, step=0.5, label="调节伴奏音量，默认为0")
      with gr.Row():
        inp_reverb = gr.Slider(minimum=0, maximum=10, value=4, step=0.5, label="混响强度", info="0为干声，4为默认值，10为宏大混响")
        inp_delay = gr.Slider(minimum=0, maximum=10, value=0, step=0.5, label="回声(延迟)效果", info="0为关闭，数值越大回声越明显")
      btn = gr.Button("一键开启AI翻唱之旅吧💕", variant="primary")
    
    with gr.Column():
      out = gr.Audio(label="AI歌手为您倾情演唱的歌曲🎶", type="filepath", interactive=False,streaming=True,)

  refresh_btn.click(refresh_models, outputs=model_dropdown,api_name=None)
  switch_btn.click(switch_model, inputs=model_dropdown, outputs=model_status)
  
  # 绑定参数中加入了 uvr_model_choice
  btn.click(convert, [inp1, inp5, inp6, inp7, model_dropdown, inp_reverb, inp_delay, uvr_model_choice], out, api_name="None")
  
  api_model_name = gr.Textbox(visible=False)
  api_output = gr.Audio(visible=False)
  gr.Button("API Convert", visible=False).click(
      convert,
      inputs=[inp1, inp5, inp6, inp7, api_model_name, inp_reverb, inp_delay, uvr_model_choice],
      outputs=[api_output],
      api_name="convert"
  )
  gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用。</center>")


print("正在初始化并加载模型列表...")
initial_models = show_model()
if initial_models:
    print(f"成功加载 {len(initial_models)} 个模型")
else:
    print("⚠️ 警告: 未能加载模型列表，请确保RVC服务正在运行")

app.queue(max_size=40, api_open=False)
app.launch(server_name="0.0.0.0", share=True, show_error=True)
