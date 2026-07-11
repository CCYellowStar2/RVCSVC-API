import os
import sys
import logging
import json

# ========================================================
# 🔥 核心修复：强制让 ONNX Runtime 借用 PyTorch 的 CUDA 库 🔥
# ========================================================
try:
    import torch
    # 获取 PyTorch 的库文件目录 (里面有 cudnn_cnn_infer_64_8.dll 等关键文件)
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    
    # 针对 Windows 系统添加 DLL 搜索路径
    if os.name == 'nt' and os.path.exists(torch_lib_path):
        # Python 3.8+ 需要用这个方法添加 DLL 目录
        try:
            os.add_dll_directory(torch_lib_path)
        except AttributeError:
            pass 
        # 同时添加到环境变量 PATH 中，双重保险
        os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ['PATH']
        print(f"✅ [GPU修复] 已将 PyTorch CUDA 库路径注入: {torch_lib_path}")
except ImportError:
    print("⚠️ 未检测到 PyTorch，无法借用 CUDA 库，可能会回退到 CPU。")
# ========================================================

import onnxruntime as ort
from audio_separator.separator import Separator

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MDXNetRoFormer:
    def __init__(self, model_path, output_dir="output"):
        # ================= GPU 检测 =================
        try:
            providers = ort.get_available_providers()
            print(f"🔍 [调试] ONNX Runtime 可用设备: {providers}")
            if 'CUDAExecutionProvider' not in providers:
                print("⚠️ [警告] 依然未检测到 CUDA！请检查你的显卡驱动或 CUDA 版本。")
                print("   (如果速度很慢，这就是原因)")
            else:
                print("🚀 [成功] RoFormer 将使用 GPU (CUDA) 进行加速！")
        except Exception as e:
            print(f"⚠️ 无法检测 GPU 状态: {e}")
        # ============================================

        # 1. 路径标准化
        self.model_path = os.path.abspath(model_path)
        self.model_dir = os.path.dirname(self.model_path)
        self.model_filename = os.path.basename(self.model_path)
        self.output_dir = os.path.abspath(output_dir)
        
        # 2. 检查配置文件
        yaml_name = os.path.splitext(self.model_filename)[0] + ".yaml"
        self.yaml_path = os.path.join(self.model_dir, yaml_name)
        
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(
                f"\n❌ 错误：找不到配置文件 {yaml_name}！\n"
                f"请确保你已经将 .yaml 文件也重命名为了：{yaml_name}"
            )

        # 3. 初始化引擎
        # 这里的 output_single_stem 只是为了防止库做多余的处理，不是必须的
        self.separator = Separator(
            log_level=logging.INFO,
            model_file_dir=self.model_dir, 
            output_dir=self.output_dir
        )

    def process_audio(self, audio_path, output_format="wav"):
        if not os.path.exists(audio_path):
            return []

        logger.info(f"正在加载模型: {self.model_filename} ...")
        
        # 显式加载模型
        self.separator.load_model(model_filename=self.model_filename)

        logger.info(f"开始分离音频: {audio_path} ...")
        output_files = self.separator.separate(audio_path)
        
        return output_files
