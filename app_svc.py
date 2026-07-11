import re, os, requests, json, torch, argparse, base64, time
from difflib import SequenceMatcher
from urllib.parse import urlparse

# --- 全局配置 ---
SVC_API_BASE = "http://127.0.0.1:7865"
TIMEOUT = 240

# --- 全局状态变量 ---
available_models, available_configs, available_diffusion_models, available_diffusion_configs = [], [], [], []
current_speaker_id = "speaker0"

# --- 参数解析 ---
parser = argparse.ArgumentParser()
parser.add_argument('--is_nohalf', action='store_true')
a = parser.parse_args()
is_half = not a.is_nohalf
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =================================================================
#               适配 so-vits-svc API 的核心函数
# =================================================================

# ========== 新增：音高优化函数 ==========
def optimize_pitch_shift(key_shift):
    """
    将升降调优化到最小调整幅度，保证最佳音质
    例如：+11 转为 -1，-10 转为 +2
    """
    if key_shift > 6:
        return key_shift - 12
    elif key_shift < -6:
        return key_shift + 12
    else:
        return key_shift
# ======================================

def find_best_fuzzy_match(source_basename, candidate_list, threshold=0.4, default_value="not_found"):
    """在候选列表中模糊查找与源名称最匹配的文件。"""
    best_score = threshold
    best_match = default_value
    for candidate_path in candidate_list:
        candidate_basename = os.path.splitext(os.path.basename(candidate_path))[0]
        score = SequenceMatcher(None, source_basename, candidate_basename).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate_path
    return best_match, best_score


def get_models_list_api():
    """
    这个函数是专门为 API 设计的。
    它只执行后端逻辑并返回纯净的 Python 列表。
    """
    # 直接调用我们已有的刷新逻辑
    models_list = refresh_models_svc()
    return models_list
    
def refresh_models_svc():
    """从 SVC API 获取并刷新所有模型和配置文件列表"""
    global available_models, available_configs, available_diffusion_models, available_diffusion_configs
    print("正在从 SVC API 刷新模型列表...")
    try:
        response = requests.post(f"{SVC_API_BASE}/run/refresh_options", json={"data": []}, timeout=TIMEOUT).json()
        
        available_models = response["data"][0]
        available_configs = response["data"][1]
        available_diffusion_models = response["data"][3]
        available_diffusion_configs = response["data"][4]
        
        print(f"✅ 成功加载 {len(available_models)} 个主模型, {len(available_diffusion_models)} 个扩散模型")
        return available_models
    except Exception as e:
        print(f"❌ 获取 SVC 模型列表失败: {e}")
        available_models, available_configs, available_diffusion_models, available_diffusion_configs = [], [], [], []
        return []

def load_svc_model(model_name: str):
    """加载指定的 SVC 模型，并使用模糊查找匹配配置文件和扩散模型"""
    global current_speaker_id
    print(f"正在请求 SVC API 加载模型: {model_name}")
    
    model_basename = os.path.splitext(model_name)[0]
    
    config_name, config_score = find_best_fuzzy_match(model_basename, available_configs, default_value="no_config")
    if config_name != "no_config":
        print(f"   模糊匹配到配置文件: {config_name} (相似度: {config_score:.2f})")
    else:
        msg = f"❌ 未找到与模型 {model_name} 匹配的 .json 配置文件"
        print(msg)
        return msg, "speaker0"

    diffusion_model_name, diff_model_score = find_best_fuzzy_match(model_basename, available_diffusion_models, default_value="no_diff")
    if diffusion_model_name != "no_diff":
        print(f"   模糊匹配到扩散模型: {diffusion_model_name} (相似度: {diff_model_score:.2f})")

    diffusion_config_name = "no_diff_config"
    if diffusion_model_name != "no_diff":
        diff_basename = os.path.splitext(diffusion_model_name)[0]
        diffusion_config_name, diff_config_score = find_best_fuzzy_match(diff_basename, available_diffusion_configs, default_value="diffusion.yaml")
        if diffusion_config_name != "no_diff_config":
            print(f"   模糊匹配到扩散配置文件: {diffusion_config_name} (相似度: {diff_config_score:.2f})")

    payload = {
        "data": [
            model_name,
            "no_clu",
            config_name,
            False,
            diffusion_model_name,
            diffusion_config_name,
            False,
            False,
            "Auto",
            "dpm-solver++",
            10,
            0,
            "nsf_hifigan_finetuned",
        ]
    }
    
    try:
        response = requests.post(f"{SVC_API_BASE}/run/load_model", json=payload, timeout=TIMEOUT).json()
        message = response["data"][0]
        speaker_info = response["data"][1]
        if isinstance(speaker_info, dict) and 'choices' in speaker_info and speaker_info['choices']:
            current_speaker_id = speaker_info['choices'][0]
            print(f"✅ 模型加载成功: {message}, 检测到说话人: {current_speaker_id}")
            return f"✅ 模型加载成功, 可用说话人: {', '.join(speaker_info['choices'])}", current_speaker_id
        else:
             print(f"✅ 模型消息: {message}, 但未检测到说话人信息。")
             return f"✅ {message}", "speaker0"
    except Exception as e:
        error_msg = f"❌ 加载 SVC 模型失败: {e}"
        print(error_msg)
        return error_msg, "speaker0"

def unload_svc_model():
    print("正在请求 SVC API 卸载模型...")
    try:
        response = requests.post(f"{SVC_API_BASE}/run/unload_model", json={"data": []}, timeout=TIMEOUT).json()
        message = response["data"][1]
        print(f"✅ {message}")
        return f"✅ {message}"
    except Exception as e:
        error_msg = f"❌ 卸载 SVC 模型失败: {e}"
        print(error_msg)
        return error_msg

def convert_svc(input_audio_path: str, speaker_id: str, key_shift: int):
    print("SVC 推理中...")
    try:
        with open(input_audio_path, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
        mime_type = "audio/wav"
        base64_audio = f"data:{mime_type};base64,{encoded_string}"
        audio_filename = os.path.basename(input_audio_path)
        audio_payload = {"name": audio_filename, "data": base64_audio}
    except Exception as e:
        raise Exception(f"读取或编码音频文件失败: {e}")
    payload = { "data": [ "wav", speaker_id, audio_payload, key_shift, False, 0, -50, 0.4, 0.5, 0, 1, 0.75, "fcpe", 0, 0.05, 100, False, False, 0 ] }
    try:
        response = requests.post(f"{SVC_API_BASE}/run/run_inference", json=payload, timeout=TIMEOUT).json()
        if "error" in response: raise Exception(response["error"])
        message, output_file_info = response["data"][0], response["data"][1]
        print(f"SVC 推理消息: {message}")
        if output_file_info and output_file_info.get("name"):
            temp_file_path_on_server = output_file_info["name"]
            download_url = f"{SVC_API_BASE}/file={temp_file_path_on_server}"
            print(f"正在从 {download_url} 下载推理结果...")
            audio_content = requests.get(download_url, timeout=TIMEOUT).content
            os.makedirs("./temp", exist_ok=True)
            local_temp_path = f"./temp/{os.path.basename(temp_file_path_on_server)}"
            with open(local_temp_path, "wb") as f: f.write(audio_content)
            print(f"推理结果已保存到: {local_temp_path}")
            return local_temp_path
        else:
            raise Exception("API 未返回有效的音频文件。")
    except Exception as e:
        print(f"❌ SVC 推理失败: {e}")
        return None

# =================================================================
#               分离模型初始化 (VR & RoFormer)
# =================================================================
from uvr5.vr import AudioPre
from msst_backend import MSST_SELECTION, separate_msst_audio
from pydub import AudioSegment
from pydub.effects import normalize
from pedalboard import Pedalboard, Compressor, Reverb, HighpassFilter, PeakFilter, LowpassFilter, PitchShift, Delay
import librosa, soundfile, gradio as gr, numpy as np
import scipy.signal
if not hasattr(scipy.signal, 'hann'):
    scipy.signal.hann = np.hanning

headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"}
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

# 1. 初始化 VR 模型
pre_fun_hp5 = AudioPre(agg=10, model_path=os.path.join(weight_uvr5_root, "5_HP-Karaoke-UVR.pth"), device=device, is_half=is_half)

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
    
def wwy_downloader(filename, uvr_choice):
    """
    下载并分离音频
    uvr_choice: "MSST BS-Roformer" 或 "VR-HP5 (默认)"
    """
    if uvr_choice == MSST_SELECTION:
        split_model = "MSST-BS-RoFormer"
        print("🚀 使用 MSST BS-Roformer 模型进行分离...")
    else:
        split_model = "UVR-HP5"
        print("🚀 使用 VR-HP5 模型进行分离...")

    audio_content = get_response(filename).content
    temp_prefixed_path = f"svc_{filename.strip()}.wav"
    with open(temp_prefixed_path, mode="wb") as f: 
        f.write(audio_content)
    
    audio_orig = AudioSegment.from_file(temp_prefixed_path)
    duration_minutes = len(audio_orig) / 60000
    print(f"原始音频时长: {duration_minutes:.2f} 分钟")
    if duration_minutes > 5:
        print("⚠️ 音频超过5分钟，正在截取前5分钟...")
        audio_orig = audio_orig[:300000]

    uvr_input_path = f"{filename.strip()}.wav"
    audio_orig.export(uvr_input_path, format="wav")

    if os.path.isfile(temp_prefixed_path):
        os.remove(temp_prefixed_path)

    output_dir = f"./output/{split_model}/{filename}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 执行分离 ==========
    if uvr_choice == MSST_SELECTION:
        final_vocal, final_inst = separate_msst_audio(
            uvr_input_path,
            output_dir,
            model_selection=uvr_choice,
        )
    else:
        # VR 模型分离
        print("分离人声伴奏 (VR-HP5)")
        pre_fun_hp5._path_audio_(uvr_input_path, output_dir, output_dir, "wav")
        final_vocal = f"{output_dir}/vocal_{filename}.wav_10.wav"
        final_inst = f"{output_dir}/instrument_{filename}.wav_10.wav"
    # ==============================
    
    if os.path.isfile(uvr_input_path):
        os.remove(uvr_input_path)
        
    return final_vocal, final_inst


def sanitize_filename(filename):
    return re.sub(r'[\\/:*?"<>|]', '', filename)

# =================================================================
#               核心转换流程 & Gradio UI
# =================================================================
def convert(song_name_src, key_shift, vocal_vol, inst_vol, model_dropdown, reverb_intensity, delay_intensity, uvr_choice="VR-HP5 (默认)"):
    """进行翻唱推理合成"""
    
    # 映射选择
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
    print(f"处理歌曲ID: {song_name_src} | 模式: {current_uvr_mode}")
    
    # 智能判断缓存路径
    if current_uvr_mode == MSST_SELECTION:
        expected_vocal = f"./output/{split_model_dir}/{song_name_src}/{song_name_src}_vocals.wav"
        expected_inst = f"./output/{split_model_dir}/{song_name_src}/{song_name_src}_other.wav"
    else:
        expected_vocal = f"./output/{split_model_dir}/{song_name_src}/vocal_{song_name_src}.wav_10.wav"
        expected_inst = f"./output/{split_model_dir}/{song_name_src}/instrument_{song_name_src}.wav_10.wav"

    if os.path.exists(expected_vocal) and os.path.exists(expected_inst):
        print(f"✅ 歌曲已缓存 ({current_uvr_mode})，跳过下载和分离")
        vocal_path = expected_vocal
        inst_path = expected_inst
    else:
        vocal_path, inst_path = wwy_downloader(song_name_src, current_uvr_mode)

    status_msg, speaker_id = load_model_ui(model_dropdown)
    inferred_audio_path = convert_svc(vocal_path, speaker_id, key_shift)
    
    if not inferred_audio_path: raise gr.Error("SVC 推理失败，请检查 SVC 服务控制台输出。")
    
    print("开始处理音频")
    audio_data, sr = librosa.load(inferred_audio_path, sr=None, mono=False)
    if audio_data.ndim == 1: audio_data = audio_data.reshape(1, -1)
    
    # ========== 混响参数计算 ==========
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
    
    # ========== 延迟效果 ==========
    if delay_intensity > 0:
        print("🎤 启用回声效果，开始准备参数...")
        try:
            print("🎵 正在检测歌曲BPM...")
            y_inst, sr_inst = librosa.load(inst_path, sr=None)
            tempo, _ = librosa.beat.beat_track(y=y_inst, sr=sr_inst)
            
            if isinstance(tempo, np.ndarray):
                actual_tempo = tempo[0]
            else:
                actual_tempo = tempo

            if actual_tempo > 0:
                print(f"✅ 检测到歌曲BPM约为: {actual_tempo:.1f}")
                delay_seconds_val = (60.0 / actual_tempo) * 0.5 
            else:
                print("⚠️ 未能检测到有效的BPM，将使用默认值。")
                delay_seconds_val = 0.5
                
        except Exception as e:
            print(f"⚠️ BPM检测失败: {type(e).__name__}: {e}，将使用默认值。")
            delay_seconds_val = 0.5
            
        delay_mix_val = (delay_intensity / 10.0) * 0.35
        effects.append(Delay(delay_seconds=delay_seconds_val, feedback=0.25, mix=delay_mix_val))
    
    effects.append(Reverb(room_size=room_size_val, damping=0.4, wet_level=wet_level_val, dry_level=dry_level_val, width=0.8))
    
    board = Pedalboard(effects)
    processed = board(audio_data, sr)
    processed_int16 = (processed.T * 32768).astype(np.int16)
    processed_audio = AudioSegment(processed_int16.tobytes(), frame_rate=sr, sample_width=2, channels=processed.shape[0])
    normalized_audio = normalize(processed_audio + vocal_vol, headroom=-1.0)
    
    # ========== 伴奏音高处理 ==========
    print("🎵 准备伴奏...")
    key_shift = optimize_pitch_shift(key_shift)
    
    if key_shift != 0 and abs(key_shift) != 12:
        print(f"🎹 正在将伴奏音高调整 {key_shift:+d} 半音以匹配人声...")
        try:
            y_inst, sr_inst = librosa.load(inst_path, sr=None)
            pitch_board = Pedalboard([PitchShift(semitones=key_shift)])
            y_shifted = pitch_board(y_inst, sr_inst)
            
            shifted_inst_path = f"temp/shifted_{song_name_src}_inst.wav"
            soundfile.write(shifted_inst_path, y_shifted, sr_inst)
            audio_inst = AudioSegment.from_file(shifted_inst_path, format="wav")
            print(f"✅ 伴奏音高调整完成")
        except Exception as e:
            print(f"⚠️ 伴奏音高调整失败: {e}")
            audio_inst = AudioSegment.from_file(inst_path, format="wav")
    else:
        audio_inst = AudioSegment.from_file(inst_path, format="wav")
        
    audio_inst = audio_inst + inst_vol
    combined_audio = normalized_audio.overlay(audio_inst)
    
    output_filename = f"temp/{sanitize_filename(song_name_src)}-SVC-AI翻唱.mp3"
    combined_audio.export(output_filename, format="MP3", bitrate="192k")
    if os.path.isfile(inferred_audio_path): os.remove(inferred_audio_path)
    print(f"已导出: {output_filename}")
    return output_filename

# --- Gradio UI 定义 ---
app = gr.Blocks()
with app:
    gr.Markdown("# <center>SVC 一键翻唱</center>")
    gr.Markdown("## <center>自动分离人声、转换、混音</center>")
    app.load(
        fn=get_models_list_api,
        outputs=gr.JSON(visible=False),
        api_name="show_model"
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                model_dropdown = gr.Dropdown(label="选择AI模型", choices=[], value=None, info="请先点击刷新加载模型列表")
                refresh_btn = gr.Button("🔄 刷新模型")
                load_btn = gr.Button("✅ 加载模型", variant="primary")
            with gr.Row():
                model_status = gr.Textbox(label="模型状态", value="请先加载模型", interactive=False)
                speaker_id_state = gr.Textbox(label="Speaker ID", value="speaker0", visible=False)
            with gr.Row():
                inp1 = gr.Textbox(label="请填写想要AI翻唱的网易云id或链接", placeholder="114514")
            
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
                inp5 = gr.Slider(-12, 12, value=0, step=1, label="歌曲人声升降调")
                inp6 = gr.Slider(-3, 3, value=0, step=0.5, label="调节人声音量(dB)")
                inp7 = gr.Slider(-3, 3, value=0, step=0.5, label="调节伴奏音量(dB)")
            with gr.Row():
                inp_reverb = gr.Slider(
                    minimum=0, maximum=10, value=4, step=0.5,
                    label="混响强度",
                    info="0为干声，4为默认值，10为宏大混响"
                )
                inp_delay = gr.Slider(
                    minimum=0, maximum=10, value=0, step=0.5,
                    label="回声(延迟)效果",
                    info="0为关闭，数值越大回声越明显"
                )
            btn = gr.Button("一键开启AI翻唱之旅吧💕", variant="primary")
        with gr.Column():
            out = gr.Audio(label="AI歌手为您倾情演唱的歌曲🎶", type="filepath", interactive=False)
    
    def refresh_models_ui():
        models_list = refresh_models_svc()
        return gr.Dropdown(choices=models_list, value=models_list[0] if models_list else "无可用模型")
    
    def load_model_ui(model_name):
        if not model_name or model_name == "无可用模型": return "❌ 请先选择一个有效的模型", "speaker0"
        status_msg, speaker_id = load_svc_model(model_name)
        return status_msg, speaker_id
        
    refresh_btn.click(refresh_models_ui, outputs=model_dropdown,api_name=None)
    load_btn.click(load_model_ui, inputs=model_dropdown, outputs=[model_status, speaker_id_state],api_name=None)
    
    # 更新点击事件，传入 uvr_model_choice
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
    gr.HTML('''<div class="footer"><p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘</p></div>''')

print("正在初始化并从 SVC API 加载模型列表...")
initial_models = refresh_models_svc()
if initial_models:
    model_dropdown.choices = initial_models
else:
    print("⚠️ 警告: 未能加载模型列表，请确保 so-vits-svc 服务正在运行")

app.queue(max_size=40, api_open=False)
app.launch(server_name="0.0.0.0",server_port=7866, share=True, show_error=True)
