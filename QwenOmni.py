from __future__ import annotations
import torch
import os
import tempfile
import io
import torchaudio
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as modelscope_snapshot_download
from PIL import Image
from pathlib import Path
import folder_paths
from qwen_omni_utils import process_mm_info
import numpy as np
import soundfile as sf
import requests
import time
from .VideoUploader import VideoUploader
import torchvision


# 模型注册表 - 存储所有支持的模型版本信息
# Model Registry - Stores information about all supported model versions
MODEL_REGISTRY = {
    "Qwen2.5-Omni-3B": {
        "repo_id": {
            "huggingface": "Qwen/Qwen2.5-Omni-3B",
            "modelscope": "qwen/Qwen2.5-Omni-3B"
        },
        "required_files": [
            "added_tokens.json", "chat_template.json", "merges.txt",
            "model.safetensors.index.json", "preprocessor_config.json", 
            "spk_dict.pt", "tokenizer.json", "vocab.json", "config.json",
            "generation_config.json", "special_tokens_map.json",
            "tokenizer_config.json",
            # 3B模型分片为3个
            # 3B model is split into 3 shards
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ],
        "test_file": "model-00003-of-00003.safetensors",
        "default": True
    },
    "Qwen2.5-Omni-7B": {
        "repo_id": {
            "huggingface": "Qwen/Qwen2.5-Omni-7B",
            "modelscope": "qwen/Qwen2.5-Omni-7B"
        },
        "required_files": [
            "added_tokens.json", "chat_template.json", "merges.txt",
            "model.safetensors.index.json", "preprocessor_config.json", 
            "spk_dict.pt", "tokenizer.json", "vocab.json", "config.json",
            "generation_config.json", "special_tokens_map.json",
            "tokenizer_config.json",
            # 7B模型有5个分片
            # 7B model has 5 shards
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
        ],
        "test_file": "model-00005-of-00005.safetensors",
        "default": False
    }
}


def check_flash_attention():
    """
    检测Flash Attention 2支持（需Ampere架构及以上）
    Check Flash Attention 2 support (requires Ampere architecture or higher)
    """
    try:
        from flash_attn import flash_attn_func
        major, _ = torch.cuda.get_device_capability()
        return major >= 8  # 仅支持计算能力8.0+的GPU
        # Only supports GPUs with compute capability 8.0+
    except ImportError:
        return False


FLASH_ATTENTION_AVAILABLE = check_flash_attention()


def init_qwen_paths(model_name):
    """
    初始化模型路径，支持动态生成不同模型版本的路径
    Initialize model paths, supporting dynamic generation of paths for different model versions
    """
    base_dir = Path(folder_paths.models_dir).resolve()
    qwen_dir = base_dir / "Qwen"
    model_dir = qwen_dir / model_name  # 使用模型名称作为子目录
    # Use model name as subdirectory
    
    # 创建目录
    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 注册到ComfyUI
    # Register to ComfyUI
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path("Qwen", str(model_dir))
    else:
        folder_paths.folder_names_and_paths["Qwen"] = ([str(model_dir)], {'.safetensors', '.bin'})
    
    print(f"模型路径已初始化: {model_dir}")
    print(f"Model path initialized: {model_dir}")
    return str(model_dir)


def test_download_speed(url):
    """
    测试下载速度，下载 5 秒
    Test download speed by downloading for 5 seconds
    """
    try:
        start_time = time.time()
        response = requests.get(url, stream=True, timeout=10)
        downloaded_size = 0
        for data in response.iter_content(chunk_size=1024):
            if time.time() - start_time > 5:
                break
            downloaded_size += len(data)
        end_time = time.time()
        speed = downloaded_size / (end_time - start_time) / 1024  # KB/s
        return speed
    except Exception as e:
        print(f"测试下载速度时出现错误: {e}")
        print(f"Error occurred while testing download speed: {e}")
        return 0


def validate_model_path(model_path, model_name):
    """
    验证模型路径的有效性和模型文件是否齐全
    Validate the validity of the model path and whether all model files are present
    """
    path_obj = Path(model_path)
    
    # 基本路径检查
    # Basic path check
    if not path_obj.is_absolute():
        print(f"错误: {model_path} 不是绝对路径")
        print(f"Error: {model_path} is not an absolute path")
        return False
    
    if not path_obj.exists():
        print(f"模型目录不存在: {model_path}")
        print(f"Model directory does not exist: {model_path}")
        return False
    
    if not path_obj.is_dir():
        print(f"错误: {model_path} 不是目录")
        print(f"Error: {model_path} is not a directory")
        return False
    
    # 检查模型文件是否齐全
    # Check if all required model files are present
    if not check_model_files_exist(model_path, model_name):
        print(f"模型文件不完整: {model_path}")
        print(f"Model files are incomplete: {model_path}")
        return False
    
    return True


def check_model_files_exist(model_dir, model_name):
    """
    检查特定模型版本所需的文件是否齐全
    Check if all files required for a specific model version are present
    """
    if model_name not in MODEL_REGISTRY:
        print(f"错误: 未知模型版本 {model_name}")
        print(f"Error: Unknown model version {model_name}")
        return False
    
    required_files = MODEL_REGISTRY[model_name]["required_files"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            return False
    return True


# 视频处理工具类
# Video Processing Utility Class
class VideoProcessor:
    def __init__(self):
        # 尝试导入torchcodec作为备选视频处理库
        # Try to import torchcodec as an alternative video processing library
        self.use_torchcodec = False
        try:
            import torchcodec
            self.use_torchcodec = True
            print("使用torchcodec进行视频处理")
            print("Using torchcodec for video processing")
        except ImportError:
            print("torchcodec不可用，使用torchvision进行视频处理")
            print("torchcodec is unavailable, using torchvision for video processing")
            # 抑制torchvision视频API弃用警告
            # Suppress torchvision video API deprecation warnings
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")
    
    def read_video(self, video_path):
        """
        读取视频文件并返回帧数据
        Read video file and return frame data
        """
        start_time = time.time()
        try:
            if self.use_torchcodec:
                # 使用torchcodec读取视频
                # Read video using torchcodec
                import torchcodec
                decoder = torchcodec.VideoDecoder(video_path)
                frames = []
                for frame in decoder:
                    frames.append(frame)
                fps = decoder.get_fps()
                total_frames = len(frames)
                frames = torch.stack(frames) if frames else torch.zeros(0)
            else:
                # 使用torchvision读取视频（弃用API）
                # Read video using torchvision (deprecated API)
                frames, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
                fps = info["video_fps"]
                total_frames = frames.shape[0]
            
            process_time = time.time() - start_time
            print(f"视频处理完成: {video_path}, 总帧数: {total_frames}, FPS: {fps:.2f}, 处理时间: {process_time:.3f}s")
            print(f"Video processing completed: {video_path}, Total frames: {total_frames}, FPS: {fps:.2f}, Processing time: {process_time:.3f}s")
            return frames, fps, total_frames
            
        except Exception as e:
            print(f"视频处理错误: {e}")
            print(f"Video processing error: {e}")
            return None, None, None


class QwenOmniCombined:
    def __init__(self):
        # 默认使用注册表中的第一个默认模型
        # Use the first default model in the registry by default
        default_model = next((name for name, info in MODEL_REGISTRY.items() if info.get("default", False)), 
                            list(MODEL_REGISTRY.keys())[0])
        
        # 重置环境变量，避免干扰
        # Reset environment variables to avoid interference
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)     

        self.current_model_name = default_model
        self.current_quantization = None  # 记录当前的量化配置
        # Record current quantization configuration
        self.model_path = init_qwen_paths(self.current_model_name)
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"模型路径: {self.model_path}")
        print(f"Model path: {self.model_path}")
        print(f"缓存路径: {self.cache_dir}")
        print(f"Cache path: {self.cache_dir}")
        
        # 验证并创建缓存目录
        # Validate and create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.video_processor = VideoProcessor()  # 初始化视频处理器
        # Initialize video processor
        self.last_generated_text = ""  # 保存上次生成的文本，用于调试
        # Save last generated text for debugging
        self.generation_stats = {"count": 0, "total_time": 0}  # 统计生成性能
        # Statistics for generation performance

    def clear_model_resources(self):
        """
        释放当前模型占用的资源
        Release resources occupied by the current model
        """
        if self.model is not None:
            print("释放当前模型占用的资源...")
            print("Releasing resources occupied by the current model...")
            del self.model, self.processor, self.tokenizer
            self.model = None
            self.processor = None
            self.tokenizer = None
            torch.cuda.empty_cache()  # 清理GPU缓存
            # Clean GPU cache
    
    def load_model(self, model_name, quantization):
        # 检查是否需要重新加载模型
        # Check if model needs to be reloaded
        if (self.model is not None and 
            self.current_model_name == model_name and 
            self.current_quantization == quantization):
            print(f"使用已加载的模型: {model_name}，量化: {quantization}")
            print(f"Using already loaded model: {model_name}, Quantization: {quantization}")
            return
        
        # 需要重新加载，先释放现有资源
        # Need to reload, release existing resources first
        self.clear_model_resources()
        
        # 更新当前模型名称和路径
        # Update current model name and path
        self.current_model_name = model_name
        self.model_path = init_qwen_paths(self.current_model_name)
        self.current_quantization = quantization
        
        # 添加CUDA可用性检查
        # Add CUDA availability check
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is required for  {model_name} model")
            # 中文提示: 运行 {model_name} 模型需要CUDA支持
            # Chinese prompt: CUDA support is required to run the {model_name} model

        # 添加警告过滤
        # Add warning filter
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")

        quant_config = None
        compute_dtype = torch.float16  # 默认使用float16
        # Use float16 by default
        if quantization == "👍 4-bit (VRAM-friendly)":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,  # 确保计算精度为float16
                # Ensure computation precision is float16
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quantization == "⚖️ 8-bit (Balanced Precision)":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype  # 确保计算精度为float16
                # Ensure computation precision is float16
            )

        # 自定义device_map，这里假设只有一个GPU，将模型尽可能放到GPU上
        # Custom device_map, assuming there's only one GPU here, place the model on the GPU as much as possible
        device_map = {"": 0} if torch.cuda.device_count() > 0 else "auto"

        # 检查模型文件是否存在且完整
        # Check if model files exist and are complete
        if not validate_model_path(self.model_path, self.current_model_name):
            print(f"检测到模型文件缺失，正在为你下载 {model_name} 模型，请稍候...")
            print(f"Missing model files detected, downloading {model_name} model for you, please wait...")
            print(f"下载将保存在: {self.model_path}")
            print(f"Download will be saved to: {self.model_path}")
            
            # 开始下载逻辑
            # Start download logic
            try:
                # 从注册表获取模型信息
                # Get model information from registry
                model_info = MODEL_REGISTRY[model_name]
                
                # 测试下载速度
                # Test download speed
                huggingface_test_url = f"https://huggingface.co/{model_info['repo_id']['huggingface']}/resolve/main/{model_info['test_file']}"
                modelscope_test_url = f"https://modelscope.cn/api/v1/models/{model_info['repo_id']['modelscope']}/repo?Revision=master&FilePath={model_info['test_file']}"
                huggingface_speed = test_download_speed(huggingface_test_url)
                modelscope_speed = test_download_speed(modelscope_test_url)

                print(f"Hugging Face下载速度: {huggingface_speed:.2f} KB/s")
                print(f"Hugging Face download speed: {huggingface_speed:.2f} KB/s")
                print(f"ModelScope下载速度: {modelscope_speed:.2f} KB/s")
                print(f"ModelScope download speed: {modelscope_speed:.2f} KB/s")

                # 根据下载速度选择优先下载源
                # Select priority download source based on download speed
                if huggingface_speed > modelscope_speed * 1.5:
                    download_sources = [
                        (snapshot_download, model_info['repo_id']['huggingface'], "Hugging Face"),
                        (modelscope_snapshot_download, model_info['repo_id']['modelscope'], "ModelScope")
                    ]
                    print("基于下载速度分析，优先尝试从Hugging Face下载")
                    print("Based on download speed analysis, trying to download from Hugging Face first")
                else:
                    download_sources = [
                        (modelscope_snapshot_download, model_info['repo_id']['modelscope'], "ModelScope"),
                        (snapshot_download, model_info['repo_id']['huggingface'], "Hugging Face")
                    ]
                    print("基于下载速度分析，优先尝试从ModelScope下载")
                    print("Based on download speed analysis, trying to download from ModelScope first")

                max_retries = 3
                success = False
                final_error = None
                used_cache_path = None

                for download_func, repo_id, source in download_sources:
                    for retry in range(max_retries):
                        try:
                            print(f"开始从 {source} 下载模型（第 {retry + 1} 次尝试）...")
                            print(f"Starting to download model from {source} (Attempt {retry + 1})...")
                            if download_func == snapshot_download:
                                cached_path = download_func(
                                    repo_id,
                                    cache_dir=self.cache_dir,
                                    ignore_patterns=["*.msgpack", "*.h5"],
                                    resume_download=True,
                                    local_files_only=False
                                )
                            else:
                                cached_path = download_func(
                                    repo_id,
                                    cache_dir=self.cache_dir,
                                    revision="master"
                                )

                            used_cache_path = cached_path  # 记录使用的缓存路径
                            # Record the cache path used
                            
                            # 将下载的模型复制到模型目录
                            # Copy the downloaded model to the model directory
                            self.copy_cached_model_to_local(cached_path, self.model_path)
                            
                            print(f"成功从 {source} 下载模型到 {self.model_path}")
                            print(f"Successfully downloaded model from {source} to {self.model_path}")
                            success = True
                            break

                        except Exception as e:
                            final_error = e  # 保存最后一个错误
                            # Save the last error
                            if retry < max_retries - 1:
                                print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，即将进行下一次尝试...")
                                print(f"Failed to download model from {source} (Attempt {retry + 1}): {e}, proceeding to next attempt...")
                            else:
                                print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，尝试其他源...")
                                print(f"Failed to download model from {source} (Attempt {retry + 1}): {e}, trying other source...")
                    if success:
                        break
                else:
                    raise RuntimeError("从所有源下载模型均失败。")
                    # 中文提示: 从所有源下载模型均失败。
                    # Chinese prompt: Failed to download model from all sources.
                
                # 下载完成后再次验证
                # Validate again after download
                if not validate_model_path(self.model_path, self.current_model_name):
                    raise RuntimeError(f"下载后模型文件仍不完整: {self.model_path}")
                    # 中文提示: 下载后模型文件仍不完整: {self.model_path}
                    # Chinese prompt: Model files are still incomplete after download: {self.model_path}
                
                print(f"模型 {model_name} 已准备就绪")
                print(f"Model {model_name} is ready")
                
            except Exception as e:
                print(f"下载模型时发生错误: {e}")
                print(f"Error occurred while downloading model: {e}")
                
                # 下载失败提示
                # Download failure prompt
                if used_cache_path:
                    print("\n⚠️ 注意：下载过程中创建了缓存文件")
                    print("\n⚠️ Attention: Cache files were created during the download process")
                    print(f"缓存路径: {used_cache_path}")
                    print(f"Cache path: {used_cache_path}")
                    print("你可以前往此路径删除缓存文件以释放硬盘空间")
                    print("You can go to this path to delete the cache files to free up disk space")
                
                raise RuntimeError(f"无法下载模型 {model_name}，请手动下载并放置到 {self.model_path}")
                # 中文提示: 无法下载模型 {model_name}，请手动下载并放置到 {self.model_path}
                # Chinese prompt: Unable to download model {model_name}, please download manually and place in {self.model_path}

        # 根据量化配置动态选择注意力实现
        # Dynamically select attention implementation based on quantization configuration
        if quant_config is not None:
            # 当使用量化时，强制使用标准注意力实现而非FlashAttention
            # When using quantization, force standard attention implementation instead of FlashAttention
            attn_impl = "sdpa"
            print("使用标准注意力实现 (sdpa) 替代FlashAttention，以兼容量化模式")
            print("Using standard attention implementation (sdpa) instead of FlashAttention to support quantization mode")
        else:
            # 非量化模式下，根据可用性选择
            # In non-quantization mode, select based on availability
            attn_impl = "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else "sdpa"

        # 设置模型精度
        # Set model precision
        model_dtype = compute_dtype if quant_config else torch.float16
        # 记录当前使用的精度
        # Record currently used precision
        precision_msg = "fp16" if model_dtype == torch.float16 else "bf16"
        print(f"使用精度: {precision_msg}")        
        print(f"Using precision: {precision_msg}")        
        # 明确设置audio部分的精度，确保与模型其他部分一致
        # Explicitly set the precision of the audio part to ensure consistency with other parts of the model
        audio_dtype = model_dtype   

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=model_dtype,
            quantization_config=quant_config,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            offload_state_dict=True,
            enable_audio_output=True,
        ).eval()

        # 编译优化（PyTorch 2.2+）
        # Compilation optimization (PyTorch 2.2+)
        if torch.__version__ >= "2.2":
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # SDP优化
        # SDP optimization
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # 修复rope_scaling配置警告 - 移至此处立即执行
        # Fix rope_scaling configuration warning - moved here for immediate execution
        if hasattr(self.model.config, "rope_scaling"):
            print("修复ROPE缩放配置...")
            print("Fixing ROPE scaling configuration...")
            if "mrope_section" in self.model.config.rope_scaling:
                self.model.config.rope_scaling["mrope_section"] = "none"  # 禁用 MROPE 优化
                # Disable MROPE optimization
            else:
                print("模型配置中没有mrope_section键，无需修复")
                print("No mrope_section key in model configuration, no fix needed")

    def copy_cached_model_to_local(self, cached_path, target_path):
        """
        将缓存的模型文件复制到目标路径
        Copy cached model files to target path
        """
        print(f"正在将模型从缓存复制到: {target_path}")
        print(f"Copying model from cache to: {target_path}")
        target_path = Path(target_path)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # 使用shutil进行递归复制
        # Use shutil for recursive copying
        import shutil
        for item in Path(cached_path).iterdir():
            if item.is_dir():
                shutil.copytree(item, target_path / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_path / item.name)
        
        # 验证复制是否成功
        # Validate if copy was successful
        if validate_model_path(target_path, self.current_model_name):
            print(f"模型已成功复制到 {target_path}")
            print(f"Model successfully copied to {target_path}")
        else:
            raise RuntimeError(f"复制后模型文件仍不完整: {target_path}")
            # 中文提示: 复制后模型文件仍不完整: {target_path}
            # Chinese prompt: Model files are still incomplete after copy: {target_path}

    def tensor_to_pil(self, image_tensor):
        """
        将图像张量转换为PIL图像
        Convert image tensor to PIL image
        """
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def preprocess_image(self, image):
        """
        预处理图像，包括尺寸调整和优化
        Preprocess image, including resizing and optimization
        """
        pil_image = self.tensor_to_pil(image)
        
        # 限制最大尺寸，避免过大的输入
        # Limit maximum size to avoid excessively large inputs
        max_res = 1024
        if max(pil_image.size) > max_res:
            pil_image.thumbnail((max_res, max_res))
        
        # 转换回张量并归一化
        # Convert back to tensor and normalize
        img_np = np.array(pil_image)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # 转回PIL图像
        # Convert back to PIL image
        pil_image = Image.fromarray((img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        return pil_image

    def preprocess_video(self, video_path):
        """
        预处理视频，包括帧提取和尺寸调整
        Preprocess video, including frame extraction and resizing
        """
        # 使用视频处理器读取视频
        # Read video using video processor
        frames, fps, total_frames = self.video_processor.read_video(video_path)
        
        if frames is None:
            print(f"无法处理视频: {video_path}")
            print(f"Unable to process video: {video_path}")
            return None, None, None
        
        # 更激进的帧数量限制
        # More aggressive frame count limit
        max_frames = 15  # 从50减少到30
        # Reduced from 50 to 30
        if total_frames > max_frames:
            # 采样帧
            # Sample frames
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            frames = frames[indices]
            print(f"视频帧数量从 {total_frames} 采样到 {len(frames)}")
            print(f"Video frame count sampled from {total_frames} to {len(frames)}")
        
        # 更小的帧尺寸
        # Smaller frame size
        resized_frames = []
        for frame in frames:
            # 转换为PIL图像
            # Convert to PIL image
            frame_pil = Image.fromarray(frame.numpy())
            # 调整大小为384x384 (原为512x512)
            # Resize to 384x384 (originally 512x512)
            frame_pil.thumbnail((384, 384))
            # 转回张量
            # Convert back to tensor
            frame_tensor = torch.from_numpy(np.array(frame_pil)).permute(2, 0, 1)
            resized_frames.append(frame_tensor)
        
        # 转换回张量
        # Convert back to tensor
        if resized_frames:
            resized_frames = torch.stack(resized_frames)
        else:
            resized_frames = torch.zeros(0)
        
        return resized_frames, fps, len(frames)  # 返回实际采样后的帧数
        # Return the actual number of sampled frames

    @torch.no_grad()
    def process(self, model_name, quantization, prompt, audio_output, audio_source, max_tokens, temperature, top_p,
                repetition_penalty, audio=None, image=None, video_path=None):
        start_time = time.time()
        
        # 确保加载正确的模型和量化配置
        # Ensure correct model and quantization configuration are loaded
        self.load_model(model_name, quantization)
        
        # 图像预处理
        # Image preprocessing
        pil_image = None
        if image is not None:
            pil_image = self.preprocess_image(image)
        
        # 音频预处理
        # Audio preprocessing
        audio_path = None
        if audio:
            try:
                # 创建临时文件并保存音频
                # Create temporary file and save audio
                with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
                    audio_path = f.name
                    waveform = audio["waveform"].squeeze(0).cpu().numpy()
                    sample_rate = audio["sample_rate"]
                    sf.write(audio_path, waveform.T, sample_rate)
            except Exception as e:
                print(f"保存音频到临时文件时出错: {e}")
                print(f"Error saving audio to temporary file: {e}")
                audio_path = None
        
        # 视频预处理
        # Video preprocessing
        video_frames = None
        if video_path:
            video_frames, video_fps, video_frames_count = self.preprocess_video(video_path)
            if video_frames is not None:
                print(f"视频已处理: {video_path}, 帧数: {video_frames_count}, FPS: {video_fps}")
                print(f"Video processed: {video_path}, Frames: {video_frames_count}, FPS: {video_fps}")
        
        # 构建对话
        # Build conversation
        SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": []}
        ]
        
        # 添加图像、音频和视频到对话
        # Add image, audio, and video to conversation
        if pil_image is not None:
            conversation[-1]["content"].append({"type": "image", "image": pil_image})
        
        use_video_audio = audio_source == "🎬 Video Built-in Audio"
        if audio_path and not use_video_audio:
            conversation[-1]["content"].append({"type": "audio", "audio": audio_path})
        
        if video_path and video_frames is not None:
            # 转换视频帧为PIL图像列表
            # Convert video frames to list of PIL images
            video_frame_list = []
            for frame in video_frames:
                frame = frame.permute(1, 2, 0).cpu().numpy() * 255
                frame = frame.astype(np.uint8)
                video_frame_list.append(Image.fromarray(frame))
            
            video_data = {
                "video": video_frame_list,
                "fps": video_fps,
                "total_frames": video_frames_count
            }
            conversation[-1]["content"].append({"type": "video", "video": video_frame_list})
            conversation[-1]["content"].append({"type": "video_data", "data": video_data})
        
        # 处理用户提示
        # Process user prompt
        user_prompt = prompt if prompt.endswith(("?", ".", "！", "。", "？", "！")) else f"{prompt} "
        conversation[-1]["content"].append({"type": "text", "text": user_prompt})
        
        # 应用聊天模板
        # Apply chat template
        input_text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # 准备处理器参数
        # Prepare processor parameters
        processor_args = {
            "text": input_text,
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": use_video_audio
        }
        
        # 调用多模态处理逻辑
        # Call multimodal processing logic
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_video_audio)
        processor_args["audio"] = audios
        processor_args["images"] = images
        processor_args["videos"] = videos
        
        # 清理不再需要的大对象
        # Clean up large objects that are no longer needed
        del video_frames, audios, images, videos
        torch.cuda.empty_cache()
        
        # 在函数开始处初始化model_inputs为None
        # Initialize model_inputs to None at the start of the function
        model_inputs = None
        
        # 将输入移至设备
        # Move inputs to device
        try:
            inputs = self.processor(**processor_args).to(self.model.device)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model_inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
                if v is not None
            }
            
            # 确保model_inputs包含所需的键
            # Ensure model_inputs contains required keys
            if "input_ids" not in model_inputs:
                raise ValueError("处理后的输入不包含'input_ids'键")
                # 中文提示: 处理后的输入不包含'input_ids'键
                # Chinese prompt: Processed input does not contain 'input_ids' key
            
        except Exception as e:
            print(f"处理输入时发生错误: {e}")
            print(f"Error occurred while processing input: {e}")
            # 这里可以添加更多的错误处理逻辑，例如返回默认值或抛出特定异常
            # More error handling logic can be added here, such as returning default values or throwing specific exceptions
            raise RuntimeError("无法处理模型输入") from e
            # 中文提示: 无法处理模型输入
            # Chinese prompt: Unable to process model input
        
        # 生成配置
        # Generation configuration
        generate_config = {
            "max_new_tokens": max(max_tokens, 10),
            "temperature": temperature,
            "do_sample": True,
            "use_cache": True,
            "return_audio": audio_output != "🔇None (No Audio)",
            "use_audio_in_video": use_video_audio,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        if generate_config["return_audio"]:
            generate_config["speaker"] = "Chelsie" if "Chelsie" in audio_output else "Ethan"
        
        # 记录GPU内存使用情况
        # Record GPU memory usage
        if torch.cuda.is_available():
            pre_forward_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"生成前GPU内存使用: {pre_forward_memory:.2f} MB")
            print(f"GPU memory usage before generation: {pre_forward_memory:.2f} MB")
        
        # 检查model_inputs是否已正确初始化
        # Check if model_inputs has been correctly initialized
        if model_inputs is None:
            raise RuntimeError("模型输入未正确初始化")
            # 中文提示: 模型输入未正确初始化
            # Chinese prompt: Model inputs not initialized correctly

        # 使用新的autocast API
        # Use new autocast API
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model.generate(**model_inputs, **generate_config)

        # 记录GPU内存使用情况
        # Record GPU memory usage
        if torch.cuda.is_available():
            post_forward_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"生成后GPU内存使用: {post_forward_memory:.2f} MB")
            print(f"GPU memory usage after generation: {post_forward_memory:.2f} MB")
            print(f"生成过程中GPU内存增加: {post_forward_memory - pre_forward_memory:.2f} MB")
            print(f"GPU memory increase during generation: {post_forward_memory - pre_forward_memory:.2f} MB")
        
        # 处理输出
        # Process outputs
        if generate_config["return_audio"]:
            text_tokens = outputs[0] if outputs[0].dim() == 2 else outputs[0].unsqueeze(0)
            audio_tensor = outputs[1]
        else:
            text_tokens = outputs if outputs.dim() == 2 else outputs.unsqueeze(0)
            audio_tensor = torch.zeros(0, 0, device=self.model.device)
        
        # 清理不再需要的大对象
        # Clean up large objects that are no longer needed
        del outputs, inputs
        torch.cuda.empty_cache()
        
        # 截取新生成的token
        # Extract newly generated tokens
        input_length = model_inputs["input_ids"].shape[1]
        text_tokens = text_tokens[:, input_length:]  # 截取新生成的token
        # Extract newly generated tokens
        
        # 解码文本
        # Decode text
        text = self.tokenizer.decode(
            text_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 保存生成的文本用于调试
        # Save generated text for debugging
        self.last_generated_text = text
        del model_inputs
        torch.cuda.empty_cache()
        
        # 清理临时文件
        # Clean up temporary files
        if audio_path:
            try:
                os.remove(audio_path)
            except Exception as e:
                print(f"删除临时音频文件时出错: {e}")
                print(f"Error deleting temporary audio file: {e}")
        
        # 处理音频输出
        # Process audio output
        if generate_config["return_audio"]:
            audio = audio_tensor
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).to(self.model.device)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
        else:
            audio = torch.zeros(0, 0, device=self.model.device)
        
        if audio.dim() == 3:
            audio = audio.mean(dim=1)
        assert audio.dim() == 2, f"Audio waveform must be 2D, got {audio.dim()}D"
        # 中文提示: 音频波形必须是2D的，得到了{audio.dim()}D
        # Chinese prompt: Audio waveform must be 2D, got {audio.dim()}D
        
        audio_output_data = {
            "waveform": audio,
            "sample_rate": 24000
        }
        
        if generate_config["return_audio"]:
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_output_data["waveform"].cpu(), 24000, format="wav")
            buffer.seek(0)
            waveform, sample_rate = torchaudio.load(buffer)
            audio_output_data = {
                "waveform": waveform.unsqueeze(0),
                "sample_rate": sample_rate
            }
        
        # 再次清理显存
        # Clean GPU memory again
        torch.cuda.empty_cache()
        
        # 计算处理时间
        # Calculate processing time
        process_time = time.time() - start_time
        self.generation_stats["count"] += 1
        self.generation_stats["total_time"] += process_time
        
        # 打印性能统计
        # Print performance statistics
        print(f"生成完成，耗时: {process_time:.2f} 秒")
        print(f"Generation completed, time taken: {process_time:.2f} seconds")
        if self.generation_stats["count"] > 0:
            avg_time = self.generation_stats["total_time"] / self.generation_stats["count"]
            print(f"平均生成时间: {avg_time:.2f} 秒/次")
            print(f"Average generation time: {avg_time:.2f} seconds/time")
        
        return (text.strip(), audio_output_data)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    list(MODEL_REGISTRY.keys()),  # 动态生成模型选项
                    # Dynamically generate model options
                    {
                        "default": next((name for name, info in MODEL_REGISTRY.items() if info.get("default", False)), 
                                       list(MODEL_REGISTRY.keys())[0]),
                        "tooltip": "Select the available model version.\n选择可用的模型版本。"
                    }
                ),
                "quantization": (
                    [
                        "👍 4-bit (VRAM-friendly)",
                        "⚖️ 8-bit (Balanced Precision)",
                        "🚫 None (Original Precision)"
                    ],
                    {
                        "default": "👍 4-bit (VRAM-friendly)",
                        "tooltip": "Select the quantization level:\n✅ 4-bit: Significantly reduces VRAM usage, suitable for resource-constrained environments.\n⚖️ 8-bit: Strikes a balance between precision and performance.\n🚫 None: Uses the original floating-point precision (requires a high-end GPU).\n\n选择量化级别：\n✅ 4位：显著减少VRAM使用，适合资源受限环境。\n⚖️ 8位：在精度和性能之间取得平衡。\n🚫 无：使用原始浮点精度（需要高端GPU）。"
                    }
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Hi!😽",
                        "multiline": True,
                        "tooltip": "Enter a text prompt, supporting Chinese and emojis. Example: 'Describe a cat in a painter's style.'\n输入文本提示，支持中文和表情符号。示例：'以画家的风格描述一只猫。'"
                    }
                ),
                "audio_output": (
                    [
                        "🔇None (No Audio)",
                        "👱‍♀️Chelsie (Female)",
                        "👨‍🦰Ethan (Male)"
                    ],
                    {
                        "default": "🔇None (No Audio)",
                        "tooltip": "Audio output options:\n🔇 Do not generate audio.\n👱‍♀️ Use the female voice Chelsie (warm tone).\n👨‍🦰 Use the male voice Ethan (calm tone).\n\n音频输出选项：\n🔇 不生成音频。\n👱‍♀️ 使用女性声音Chelsie（温暖语调）。\n👨‍🦰 使用男性声音Ethan（平静语调）。"
                    }
                ),
                "audio_source": (
                    [
                        "🎧 Separate Audio Input",
                        "🎬 Video Built-in Audio"
                    ],
                    {
                        "default": "🎧 Separate Audio Input",
                        "display": "radio",
                        "tooltip": "Select audio source: Use video's built-in audio track (priority) / Input a separate audio file (external audio)\n选择音频源：使用视频内置音轨（优先）/输入单独的音频文件（外部音频）"
                    }
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 132,
                        "min": 64,
                        "max": 2048,
                        "step": 16,
                        "display": "slider",
                        "tooltip": "Control the maximum length of the generated text (in tokens). \nGenerally, 100 tokens correspond to approximately 50 - 100 Chinese characters or 67 - 100 English words, but the actual number may vary depending on the text content and the model's tokenization strategy. \nRecommended range: 64 - 512.\n\n控制生成文本的最大长度（以token为单位）。\n一般来说，100个token约对应50-100个汉字或67-100个英文单词，但实际数量可能因文本内容和模型的分词策略而异。\n推荐范围：64-512。"
                    }
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                        "tooltip": "Control the generation diversity:\n▫️ 0.1 - 0.3: Generate structured/technical content.\n▫️ 0.5 - 0.7: Balance creativity and logic.\n▫️ 0.8 - 1.0: High degree of freedom (may produce incoherent content).\n\n控制生成多样性：\n▫️ 0.1-0.3：生成结构化/技术性内容。\n▫️ 0.5-0.7：平衡创造性和逻辑性。\n▫️ 0.8-1.0：高度自由（可能产生不连贯内容）。"
                    }
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Nucleus sampling threshold:\n▪️ Close to 1.0: Retain more candidate words (more random).\n▪️ 0.5 - 0.8: Balance quality and diversity.\n▪️ Below 0.3: Generate more conservative content.\n\n核采样阈值：\n▪️ 接近1.0：保留更多候选词（更随机）。\n▪️ 0.5-0.8：平衡质量和多样性。\n▪️ 低于0.3：生成更保守的内容。"
                    }
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Control of repeated content:\n⚠️ 1.0: Default behavior.\n⚠️ >1.0 (Recommended 1.2): Suppress repeated phrases.\n⚠️ <1.0 (Recommended 0.8): Encourage repeated emphasis.\n\n控制重复内容：\n⚠️ 1.0：默认行为。\n⚠️ >1.0（推荐1.2）：抑制重复短语。\n⚠️ <1.0（推荐0.8）：鼓励重复强调。"
                    }
                )
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Upload a reference image (supports PNG/JPG), and the model will adjust the generation result based on the image content.\n上传参考图像（支持PNG/JPG），模型将根据图像内容调整生成结果。"
                    }
                ),
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Upload an audio file (supports MP3/WAV), and the model will analyze the audio content and generate relevant responses.\n上传音频文件（支持MP3/WAV），模型将分析音频内容并生成相关响应。"
                    }
                ),
                "video_path": (
                    "VIDEO_PATH",
                    {
                        "tooltip": "Enter the video file  (supports MP4/WEBM), and the model will extract visual features to assist in generation.\n输入视频文件路径（支持MP4/WEBM），模型将提取视觉特征辅助生成。"
                    }
                )
            }
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("text", "audio")
    FUNCTION = "process"
    CATEGORY = "🐼QwenOmni"    


NODE_CLASS_MAPPINGS = {
    "VideoUploader": VideoUploader,
    "QwenOmniCombined": QwenOmniCombined
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoUploader": "Video Uploader🐼",
    "QwenOmniCombined": "Qwen Omni Combined🐼"
}
