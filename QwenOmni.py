import torch
import os
import io
from io import BytesIO
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import folder_paths
from qwen_omni_utils import process_mm_info
import numpy as np
import soundfile as sf
import torchaudio
device = "cuda"

def check_flash_attention():
    """Check if flash attention 2 is available"""
    try:
        from flash_attn import flash_attn_func
        return True
    except ImportError:
        return False

FLASH_ATTENTION_AVAILABLE = check_flash_attention()

def init_qwen_paths():
    """动态初始化模型路径"""
    qwen_base = Path(folder_paths.models_dir) / "Qwen"
    qwen_model_dir = qwen_base / "Qwen2.5-Omni-7B"
    
    # 创建必要目录
    qwen_model_dir.mkdir(parents=True, exist_ok=True)
    
    # 注册路径到系统
    if not hasattr(folder_paths, "add_model_folder_path"):
        # 兼容旧版本手动注册
        if "Qwen" not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["Qwen"] = ([str(qwen_model_dir)], {'.safetensors', '.bin'})
    else:
        folder_paths.add_model_folder_path("Qwen", str(qwen_model_dir))
    
    return str(qwen_model_dir)


class LoadQwenOmniModel:
    def __init__(self):
        self.model_path = init_qwen_paths()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
            }
        }

    RETURN_TYPES = ("QWENOMNI", "OMNIPROCESSOR")
    RETURN_NAMES = ("Qwen_Omni", "processor")
    FUNCTION = "load_model"
    CATEGORY = "🐼QwenOmni"
  
    def load_model(self):
        # 添加Flash Attention支持判断
        attn_implementation = "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else None
        # 添加量化配置
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # 改用FP16加速计算
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,        # 打开or关闭双重量化
            llm_int8_threshold=6.0,
        )

        


        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # 使用更高效的BF16格式
            quantization_config=quant_config,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            use_safetensors=True,  # 启用更快的safetensors格式
            offload_state_dict=True,  # 优化显存分配
            # max_memory={0: "14GiB", "cpu": "64GiB"},  # 精确控制显存分配
            enable_audio_output=True
        ).eval()

        
        # 预加载模型到显存
        
        processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)

         # 新增预热步骤 ▼▼▼
        with torch.no_grad():
            warmup_prompt = "generating text and speech"
            inputs = processor(
                text=warmup_prompt,
                return_tensors="pt"
            ).to(model.device)
            model.generate(
                inputs.input_ids,
                max_new_tokens=1,
                do_sample=False
            )

        return model, processor


class QwenOmniParser:
    def __init__(self):
        self.model = None
        self.processor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWENOMNI",),
                "processor": ("OMNIPROCESSOR",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail", 
                    "multiline": True
                }),
                "system_prompt": ("STRING", {
                    "default": "generating text",
                    "multiline": True,
                    "hidden": True
                }),
                "max_tokens": ("INT", {
                    "default": 128, 
                    "min": 32, 
                    "max": 512
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
                # 新增音频生成开关
                "enable_audio": ("BOOLEAN", {"default": True}),
                "voice_type": ([
                    "Chelsie",
                    "Ethan",
                    ], {"default": "Chelsie"}),
            }
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("text", "audio")
    FUNCTION = "analyze_image"
    CATEGORY = "🐼QwenOmni"

    def tensor_to_pil(self, image_tensor):
        """优化图像张量转换"""
        # 处理批次维度 [B x H x W x C]
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
            
        # 数值范围转换 [0-1] => [0-255]
        image_np = image_tensor.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
            
        return Image.fromarray(image_np)
    

    def build_multimodal_inputs(self, pil_image, prompt, system_prompt):
        """构建多模态输入结构"""
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    @torch.no_grad()
    def analyze_image(self, model, processor,image, prompt, system_prompt, max_tokens, temperature, enable_audio, voice_type):
        # 转换输入格式
        pil_image = self.tensor_to_pil(image)
               # 动态替换系统提示 ▼▼▼
        official_system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        actual_system_prompt = official_system_prompt if enable_audio else system_prompt
        
        conversation = self.build_multimodal_inputs(
            pil_image, 
            prompt,
            actual_system_prompt
        )
        
        # # 构建多模态对话
        # conversation = self.build_multimodal_inputs(pil_image, prompt, system_prompt)
        
        # 预处理多模态数据
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        audios, images, videos = process_mm_info([conversation], use_audio_in_video=False)
        
        # 准备模型输入
        inputs = processor(
            text=text,
            images=images,
            audio=audios,
            videos=videos,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        # # 生成参数配置
        generate_config = {
        "max_new_tokens": max_tokens,
        "do_sample": False,
        "temperature": temperature,
        "use_cache": True,
        "past_key_values": None, 
        "return_audio": enable_audio,  # 连接输入参数
        "speaker": voice_type  # 指定发音人
        }
        text_ids, audio = model.generate(**inputs,**generate_config)
        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return (text[0], audio)
        

class SaveQwenOmniAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {"default": "output.wav"}),
                "samplerate": ("INT", {"default": 23000}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "🐼QwenOmni"    
    def save_audio(self, audio, filename, samplerate):
        # 获取ComfyUI的输出目录
        output_dir = folder_paths.get_output_directory()
        
        # 确保文件名不包含路径（防止目录注入）
        filename = os.path.basename(filename)
        
        # 构建完整保存路径
        full_path = os.path.join(output_dir, filename)
        
        # 创建目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存音频文件
        sf.write(
            full_path,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=samplerate,
        )
        return ()



NODE_CLASS_MAPPINGS = {
    "LoadQwenOmniModel": LoadQwenOmniModel,
    "QwenOmniParser": QwenOmniParser,
    "SaveQwenOmniAudio": SaveQwenOmniAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadQwenOmniModel": "Load QwenOmni Model🐼",
    "QwenOmniParser": "QwenOmni Parser🐼",
    "SaveQwenOmniAudio": "Save QwenOmni Audio🐼",
}