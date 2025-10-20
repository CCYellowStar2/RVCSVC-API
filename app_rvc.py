import re, os
import requests
import json
import torch
import shutil
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

def download_file_openxlab(url, destination):
    # 检查目标文件是否已经存在
    if os.path.exists(destination):
        print("File already exists, skipping download.")
        return
    else:
        print(" start download... "+destination)  
        
    # 获取目标文件的目录部分
    directory = os.path.dirname(destination)
    
    # 确保目标文件夹存在
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(destination, 'wb') as f:
                    f.write(response.content)
                print("File downloaded successfully!")
                break
            else:
                print(f"Failed to download file. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error occurred: {e}")
        
        print("Retrying in 5 seconds...")
        time.sleep(5)
        
url_uvr = "https://modelscope.cn/api/v1/models/CCYellowStar/5_HP-Karaoke-UVR/repo?Revision=master&FilePath=5_HP-Karaoke-UVR.pth"
destination_uvr = "uvr5/uvr_model/5_HP-Karaoke-UVR.pth"
download_file_openxlab(url_uvr, destination_uvr)

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

def change_model(model):
  """切换模型"""
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
  """获取可用模型列表"""
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
    
    # 提取模型名（去掉扩展名）
    model_name = os.path.splitext(model)[0].lower()
    
    # 计算每个 index 文件的相似度
    best_match = None
    best_score = 0
    threshold = 0.4
    
    for index_path in index:
        # 提取 index 文件名（去掉路径和扩展名）
        index_name = os.path.splitext(os.path.basename(index_path))[0].lower()
        
        # 计算相似度
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
weight_uvr5_root = "uvr5/uvr_model"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

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

split_model = "UVR-HP5"
  

# 替换这个函数
def wwy_downloader(
    filename,
    split_model
):
    audio_content = get_response(filename).content
    # 1. 下载到带前缀的临时文件，避免冲突
    temp_prefixed_path = "rvc_" + filename.strip() + ".wav"
    with open(temp_prefixed_path, mode="wb") as f:
        f.write(audio_content)
    
    # 2. 从带前缀的文件加载和处理音频
    audio_orig = AudioSegment.from_file(temp_prefixed_path)
    duration_minutes = len(audio_orig) / 60000
    print(f"原始音频时长: {duration_minutes:.2f} 分钟")
    if duration_minutes > 5:
        print("⚠️ 音频超过5分钟，正在截取前5分钟...")
        audio_orig = audio_orig[:300000]
    
    # 3. 【关键修复】在调用UVR之前，将处理好的音频导出为UVR期望的、不带前缀的文件名
    uvr_input_path = filename.strip() + ".wav"
    audio_orig.export(uvr_input_path, format="wav")
    
    # 4. 删除带前缀的临时文件，我们不再需要它了
    if os.path.isfile(temp_prefixed_path):
        os.remove(temp_prefixed_path)

    # 5. 调用UVR，现在它会生成正确的文件名了
    os.makedirs(f"./output/{split_model}/{filename}/", exist_ok=True)
    pre_fun = pre_fun_hp5
    print("分离人声伴奏")
    pre_fun._path_audio_(uvr_input_path, f"./output/{split_model}/{filename}/", f"./output/{split_model}/{filename}/", "wav")
    
    # 6. 删除UVR用过的输入文件
    if os.path.isfile(uvr_input_path):
        os.remove(uvr_input_path)

    # 7. 返回正确的文件路径，现在这个文件肯定存在
    return f"./output/{split_model}/{filename}/vocal_{filename}.wav_10.wav", f"./output/{split_model}/{filename}/instrument_{filename}.wav_10.wav"



def convert(song_name_src, key_shift, vocal_vol, inst_vol, model_dropdown):
  """进行翻唱推理合成"""
  split_model = "UVR-HP5"
  if not song_name_src: raise gr.Error("请输入歌曲ID或链接！")
  
  if song_name_src.startswith("http"):
    try: song_name_src = song_name_src.split('id=')[1].split('&')[0]
    except IndexError: raise gr.Error("无效的网易云链接格式！")
  
  song_name_src = song_name_src.strip()
  print(f"处理歌曲ID: {song_name_src}")
  
  audio_rvc_path = os.path.join(SCRIPT_DIR, "audio_rvc.wav")
  vocal_cache_path = f"./output/{split_model}/{song_name_src}/vocal_{song_name_src}.wav_10.wav"
  
  # === 修复：检查文件是否真实存在，而不只是检查目录 ===
  if os.path.isfile(vocal_cache_path):
    print("✅ 歌曲已缓存，跳过下载")
    audio, sr = librosa.load(vocal_cache_path, sr=44100, mono=True)
    soundfile.write(audio_rvc_path, audio, sr)
  else:
    print("📥 未找到缓存，开始下载和分离")
    audio_rvc, sr_src = librosa.load(wwy_downloader(song_name_src, split_model)[0], sr=44100, mono=True)
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
  print(response["data"][0])

  if data:
    print("🎛️ 开始处理音频")
    os.makedirs("./temp", exist_ok=True)

    audio_data, sr = librosa.load(data, sr=None, mono=False)

    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(1, -1)

    from pedalboard import Pedalboard, Compressor, Reverb, HighpassFilter, PeakFilter, LowpassFilter

    board = Pedalboard([
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
        Reverb(
            room_size=0.50,
            damping=0.4,
            wet_level=0.3,
            dry_level=0.7,
            width=0.7
        )
    ])

    processed = board(audio_data, sr)
    processed_int16 = (processed.T * 32768).astype(np.int16)
    processed_audio = AudioSegment(
        processed_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=processed.shape[0]
    )
    
    audio_vocal_adjusted = processed_audio + vocal_vol
    normalized_audio = normalize(audio_vocal_adjusted, headroom=-1.0)
    
    print("🎵 混合伴奏...")
    audio_inst = AudioSegment.from_file(
        f"output/{split_model}/{song_name_src}/instrument_{song_name_src}.wav_10.wav",
        format="wav"
    )
    audio_inst = audio_inst + inst_vol
    combined_audio = normalized_audio.overlay(audio_inst)

    print("💾 导出最终文件...")
    output_path = f"temp/{sanitize_filename(song_name_src)}-RVC-AI翻唱.mp3"
    combined_audio.export(
        output_path,
        format="MP3",
        bitrate="192k"
    )
    
    if os.path.isfile(data):
      os.remove(data)
    
    print(f"✅ 已导出: {output_path}")
    return output_path

def sanitize_filename(filename):
    # 定义 Windows 禁止的字符： \ / : * ? " < > |
    # 使用正则表达式移除这些字符
    clean_name = re.sub(r'[\\/:*?"<>|]', '', filename)
    return clean_name

def refresh_models():
    """刷新模型列表的回调函数"""
    models_list = show_model()
    if models_list:
        return gr.Dropdown(choices=models_list, value=models_list[0] if models_list else None)
    else:
        return gr.Dropdown(choices=["无可用模型"], value="无可用模型")

def switch_model(model_name):
    """切换模型的回调函数 - 返回状态信息"""
    if not model_name or model_name == "无可用模型":
        return "❌ 请先选择一个有效的模型"
    result = change_model(model_name)
    return result
    
app = gr.Blocks()

with app:
  gr.Markdown("# <center>RVC一键翻唱、重磅更新！</center>")
  gr.Markdown("## 自动分离人声翻唱并合并，自动混音！</center>")
  
  with gr.Row():
    with gr.Column():
      # 模型选择区域
      with gr.Row():
        model_dropdown = gr.Dropdown(
          label="选择AI模型", 
          choices=[], 
          value=None,
          info="请先点击刷新加载模型列表"
        )
        refresh_btn = gr.Button("🔄 刷新", size="sm")
        switch_btn = gr.Button("✨ 切换模型", size="sm", variant="primary")
      with gr.Row(visible=False):  # 隐藏这个功能
          models_json = gr.JSON()
          get_models_btn = gr.Button("获取模型列表", visible=False)
          get_models_btn.click(show_model, outputs=models_json)
      # 模型状态显示
      with gr.Row():
        model_status = gr.Textbox(label="模型状态", value="请选择模型", interactive=False)
      
      with gr.Row():
        inp1 = gr.Textbox(label="请填写想要AI翻唱的网易云id或链接", placeholder="114514", info="直接填写网易云id或链接")
      
      with gr.Row():
        inp5 = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="歌曲人声升降调", info="默认为0，+2为升高2个key，以此类推")
        inp6 = gr.Slider(minimum=-3, maximum=3, value=0, step=1, label="调节人声音量，默认为0")
        inp7 = gr.Slider(minimum=-3, maximum=3, value=0, step=1, label="调节伴奏音量，默认为0")
      
      btn = gr.Button("一键开启AI翻唱之旅吧💕", variant="primary")
    
    with gr.Column():
      out = gr.Audio(label="AI歌手为您倾情演唱的歌曲🎶", type="filepath", interactive=False,streaming=True,)

  # 绑定事件
  refresh_btn.click(refresh_models, outputs=model_dropdown,api_name=None)
  switch_btn.click(switch_model, inputs=model_dropdown, outputs=model_status)
  btn.click(convert, [inp1, inp5, inp6, inp7,model_dropdown], out, api_name="None")
  api_model_name = gr.Textbox(visible=False)
  api_output = gr.Audio(visible=False)
  gr.Button("API Convert", visible=False).click(
      convert,
      inputs=[inp1, inp5, inp6, inp7, api_model_name],
      outputs=[api_output],
      api_name="convert"  # 这个才是外部API要调用的端点
  )
  gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用。</center>")
  gr.HTML('''
      <div class="footer">
                  <p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘
                  </p>
      </div>
  ''')


print("正在初始化并加载模型列表...")
initial_models = show_model()
if initial_models:
    print(f"成功加载 {len(initial_models)} 个模型")
else:
    print("⚠️ 警告: 未能加载模型列表，请确保RVC服务正在运行")


app.queue(max_size=40, api_open=False)
app.launch(server_name="0.0.0.0", share=True, show_error=True)

