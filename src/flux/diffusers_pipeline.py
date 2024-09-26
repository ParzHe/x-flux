from datetime import datetime
import os
import time
import torch
import gradio as gr
import peft

from diffusers import FluxPipeline, AutoencoderKL, FluxControlNetPipeline, FluxControlNetModel, DiffusionPipeline
from diffusers.models import FluxMultiControlNetModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel

from dataclasses import dataclass
from .model import Flux, FluxParams
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from src.utils import get_gpu_mem_info,flush,flush_without_peak,save_images_with_prompt,save_images

@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    models_dir: str | None
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    repo_id_ae: str | None

configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        models_dir=os.getenv("FLUX_DEV_DIR"),
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-merged": ModelSpec(
        repo_id="sayakpaul/FLUX.1-merged",
        repo_id_ae="sayakpaul/FLUX.1-merged",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        models_dir=os.getenv("FLUX_MERGED_DIR"),
        ckpt_path=os.getenv("FLUX_MERGED"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fp8": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux-dev-fp8.safetensors",
        repo_ae="ae.safetensors",
        models_dir=os.getenv("FLUX_DEV_FP8_DIR"),
        ckpt_path=os.getenv("FLUX_DEV_FP8"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        models_dir=os.getenv("FLUX_SCHNELL_DIR"),
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}

class DiffusersFluxPipeline:
    def __init__(self, model_type: str, device: str | torch.device = "cuda", offload: bool = False):
        print("初始化 Diffusers 管线中...")
        gr.Info("初始化 Diffusers 管线中...",duration=5)
        
        self.model_type=model_type
        self.models_dir = configs[model_type].models_dir
        self.device = device
        self.offload = offload
        self.torch_dtype=torch.bfloat16
        
        flush()
        self.gpu_mem_total, self.gpu_mem_used, self.gpu_mem_free = get_gpu_mem_info(gpu_id=0)
        
        self.control_pipe=False
        
        self.is_loaded_control=False
        self.loaded_control=None
        
        self.is_loaded_lora=False
        self.loaded_lora=None
        self.loaded_lora_scale=None
        
        self.first=True
        
        if not self.offload:
            self.pipeline=FluxPipeline.from_pretrained(self.models_dir, torch_dtype=self.torch_dtype)
            self.pipeline.to(self.device)
        else:
            self.text_encoder = CLIPTextModel.from_pretrained(self.models_dir,subfolder="text_encoder",torch_dtype=self.torch_dtype)
            self.text_encoder_2 = T5EncoderModel.from_pretrained(self.models_dir,subfolder="text_encoder_2",torch_dtype=self.torch_dtype)
            self.tokenizer = CLIPTokenizer.from_pretrained(self.models_dir, subfolder="tokenizer")
            self.tokenizer_2 = T5TokenizerFast.from_pretrained(self.models_dir, subfolder="tokenizer_2")
            self.pipeline=FluxPipeline.from_pretrained(
                self.models_dir,
                text_encoder=self.text_encoder,
                textencoder_2=self.text_encoder_2,
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                transformer=None,
                vae=None,
                revision="refs/pr/7",
            )
            self.pipeline.to(self.device)
        
        print("初始化 Diffusers 管线完成。")
        gr.Info("初始化 Diffusers 管线完成。",duration=2)
    
    @torch.inference_mode()
    def gradio_generate(self, prompt, image_prompt, controlnet_image, width, height, guidance,
                        num_steps, seed, true_gs, 
                        is_ip_enable, ip_scale, neg_ip_scale, neg_prompt,
                        neg_image_prompt, timestep_to_start_cfg, 
                        is_contronet_enable, control_type, control_weight,
                        is_lora_enable, lora_weight, local_path, lora_local_path, ip_local_path, output_dir,
                    ):
        seed = int(seed)
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()
        generator = torch.Generator().manual_seed(seed)
        pipe=None
        start_time = time.time()
        
        def lora_component(is_enable,lora_path,lora_scale):
            if is_enable:
                if lora_path != self.loaded_lora:
                    if not os.path.isfile(lora_path):
                        gr.Error("这个模型不存在, 请输入正确的模型地址")
                            
                    if self.loaded_lora != None:
                        print("Unloading lora...")
                        self.pipeline.unfuse_lora()
                        self.pipeline.unload_lora_weights()
                        print("Successfully unloaded!")
                                
                    self.pipeline.load_lora_weights(lora_path)
                    self.pipeline.fuse_lora(lora_scale=lora_scale)
                    self.pipeline.to(self.device)
                elif lora_scale!=self.loaded_lora_scale:
                    print("Change LoRA scale...")
                    self.pipeline.unfuse_lora()
                    self.pipeline.fuse_lora(lora_scale=lora_scale)
                    print("Change successfully")
                        
                self.is_loaded_lora = True
                self.loaded_lora=lora_path
                self.loaded_lora_scale=lora_scale
            elif self.loaded_lora != None:
                self.pipeline.unfuse_lora()
                self.pipeline.unload_lora_weights()
                self.is_loaded_lora = False
                self.loaded_lora=None
                self.loaded_lora_scale=None
            
        if not self.offload:
            # flush()
            if is_contronet_enable and controlnet_image is not None:
                self.first = False
                
                if not self.control_pipe or local_path != self.loaded_control:
                    
                    self.control_pipe = True
                    
                    self.is_loaded_control = True
                    self.loaded_control = None
                    
                    self.is_loaded_lora = False
                    self.loaded_lora=None
                    self.loaded_lora_scale=None
                    
                    del self.pipeline
                    if pipe is not None:
                        del pipe
                    
                    flush_without_peak()
                
                    self.loaded_control=local_path
                    controlnet_a = FluxControlNetModel.from_pretrained(local_path, torch_dtype=self.torch_dtype)
                    
                    # controlnet=FluxMultiControlNetModel([controlnet_a])
                    self.pipeline = FluxControlNetPipeline.from_pretrained(self.models_dir, controlnet=controlnet_a, torch_dtype=torch.bfloat16)
                        
                    self.pipeline.to(self.device)
                
                control_image=load_image(controlnet_image) 
                control_mode = control_weight
                controlnet_conditioning_scale=0.5
                
                lora_component(is_enable=is_lora_enable,lora_path=lora_local_path,lora_scale=lora_weight)
                    
                images=self.pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_steps,
                    guidance_scale=true_gs,
                    control_image=[control_image],
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    control_mode=control_mode,
                    num_images_per_prompt=1,
                    generator=generator,
                    max_sequence_length= 256 if self.model_type=="flux-schnell" else 512,
                ).images          
            
            else:
                if self.control_pipe is True:
                    self.control_pipe = False
                    
                    self.is_loaded_control=False
                    self.loaded_control=None
                    
                    self.is_loaded_lora = False
                    self.loaded_lora=None
                    self.loaded_lora_scale=None
                    
                    if self.pipeline is not None:
                        del self.pipeline
                    if pipe is not None:
                        del pipe
                    
                    flush()
                    
                    if self.first is False:
                        self.pipeline = FluxPipeline.from_pretrained(self.models_dir, torch_dtype=self.torch_dtype)
                        self.pipeline.to(self.device)
                        self.first = True
                        
                lora_component(is_enable=is_lora_enable,lora_path=lora_local_path,lora_scale=lora_weight)
                
                images = self.pipeline (
                    prompt=prompt, 
                    height=height, width=width,
                    num_inference_steps=num_steps, 
                    guidance_scale=true_gs,
                    generator=generator,
                    max_sequence_length= 512 if self.model_type=="flux-dev" else 256,
                ).images
        else:
            if self.first is False:
                del self.pipeline
                if pipe is not None:
                    del pipe
                flush()
                self.text_encoder = CLIPTextModel.from_pretrained(self.models_dir,subfolder="text_encoder",torch_dtype=self.torch_dtype)
                self.text_encoder_2 = T5EncoderModel.from_pretrained(self.models_dir,subfolder="text_encoder_2",torch_dtype=self.torch_dtype)
                self.tokenizer = CLIPTokenizer.from_pretrained(self.models_dir, subfolder="tokenizer")
                self.tokenizer_2 = T5TokenizerFast.from_pretrained(self.models_dir, subfolder="tokenizer_2")
            
            if is_contronet_enable and controlnet_image is not None:
                if self.first is True:
                    del self.pipeline
                    flush()
                controlnet_a = FluxControlNetModel.from_pretrained(local_path, torch_dtype=self.torch_dtype)
                # controlnet=FluxMultiControlNetModel([controlnet_a])
                self.pipeline = FluxControlNetPipeline.from_pretrained(
                    self.models_dir,
                    text_encoder=self.text_encoder,
                    textencoder_2=self.text_encoder_2,
                    tokenizer=self.tokenizer,
                    tokenizer_2=self.tokenizer_2,
                    transformer=None,
                    vae=None,
                    revision="refs/pr/7",
                    torch_dtype=self.torch_dtype
                ).to(self.device)
            elif self.first is not True:
                self.pipeline = FluxPipeline.from_pretrained(
                    self.models_dir,
                    text_encoder=self.text_encoder,
                    textencoder_2=self.text_encoder_2,
                    tokenizer=self.tokenizer,
                    tokenizer_2=self.tokenizer_2,
                    transformer=None,
                    vae=None,
                    revision="refs/pr/7",
                    torch_dtype=self.torch_dtype
                ).to(self.device)
            
            with torch.no_grad():
                print("Encoding prompts.")
                gr.Info("编码Prompt中...",duration=3)
                prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
                    prompt=prompt, prompt_2=None, max_sequence_length=256
                )
            
            self.first = False
            
            print("Type of the prompt embeds:",type(prompt_embeds))
                    
            del self.text_encoder
            del self.text_encoder_2
            del self.tokenizer
            del self.tokenizer_2
            del self.pipeline
                
            flush_without_peak()
            
            if is_contronet_enable and controlnet_image is not None:
                pipe = FluxControlNetPipeline.from_pretrained(
                    self.models_dir,
                    text_encoder=None,
                    text_encoder_2=None,
                    tokenizer=None,
                    tokenizer_2=None,
                    vae=None,
                    controlnet=controlnet_a,
                    torch_dtype=self.torch_dtype,
                ).to(self.device)
            else:        
                pipe = FluxPipeline.from_pretrained(
                    self.models_dir,
                    text_encoder=None,
                    text_encoder_2=None,
                    tokenizer=None,
                    tokenizer_2=None,
                    vae=None,
                    torch_dtype=self.torch_dtype,
                ).to(self.device)
            
            if is_lora_enable:
                pipe.load_lora_weights(lora_local_path,torch_dtype=self.torch_dtype)
                pipe.fuse_lora(lora_scale=lora_weight,torch_dtype=self.torch_dtype)
            
            print("Running denoising...")
            gr.Info("开始降噪...",duration=3)
            
            if is_contronet_enable and controlnet_image is not None:
                control_image=load_image(controlnet_image) 
                controlnet_conditioning_scale=0.5
                control_mode = control_weight
                latents = pipe(
                    prompt_embeds=prompt_embeds.to(torch.bfloat16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(torch.bfloat16),
                    height=height,
                    width=width,
                    num_inference_steps=num_steps,
                    guidance_scale=true_gs,
                    control_image=[control_image],
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    control_mode=control_mode,
                    num_images_per_prompt=1,
                    generator=generator,
                    output_type="latent",
                ).images
            else:
                latents = pipe(
                    prompt_embeds=prompt_embeds.to(torch.bfloat16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(torch.bfloat16),
                    height=height,
                    width=width,
                    num_inference_steps=num_steps,
                    guidance_scale=true_gs,
                    num_images_per_prompt=1,
                    generator=generator,
                    output_type="latent",
                ).images
            print(f"Latents Shape: {latents.shape}")
                    
            del pipe.transformer
            del pipe
                    
            flush_without_peak()
                    
            vae = AutoencoderKL.from_pretrained(self.models_dir, revision="refs/pr/1", subfolder="vae", torch_dtype=self.torch_dtype).to(
                    self.device
            )
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
            image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
                
            with torch.no_grad():
                print("Running decoding.")
                gr.Info("解码中...",duration=3)
                    
                latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
                latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

                images = vae.decode(latents, return_dict=False)[0]
                images = image_processor.postprocess(images, output_type="pil")  
        
        # 计算用时和峰值显存占用
        elapsed_time = time.time() - start_time
        max_vram_usage = torch.cuda.max_memory_allocated() / 1024 / 1024 /1024 # GB 
                
        timestamp_after_generation = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
                
        print(f"推理已完成！")
        print("生成所耗费的时间：", elapsed_time, "s")
        print("峰值显存占用：", max_vram_usage, "GB")
                
        saved_paths=None
        whether_save_prompt=True
        if whether_save_prompt:
            _,saved_paths=save_images_with_prompt(
                prompt=prompt,
                seed=seed,
                guidance_scale=true_gs,
                checkpoint=self.models_dir,
                width = width, height = height,
                num_inference_steps = num_steps,
                max_memory_usage=max_vram_usage,
                generation_time=elapsed_time,
                images=images,
                timestamp=timestamp_after_generation,
                output_folder=output_dir,
            )
        else:
            _,saved_paths=save_images(images,timestamp_after_generation,output_folder=output_dir)
                
        return images[0],saved_paths[0]       