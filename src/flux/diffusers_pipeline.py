import os
import torch
import gradio as gr

from diffusers import FluxPipeline, AutoencoderKL, FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel

from dataclasses import dataclass
from .model import Flux, FluxParams
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from src.utils import get_gpu_mem_info,flush

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
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
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
        self.models_dir = configs[model_type].models_dir
        self.device = device
        self.offload = offload
        self.torch_dtype=torch.bfloat16
        
        flush()
        self.gpu_mem_total, self.gpu_mem_used, self.gpu_mem_free = get_gpu_mem_info(gpu_id=0)
        
        self.first=True
        if not self.offload:
            self.pipeline=FluxPipeline.from_pretrained(self.models_dir, torch_dtype=self.torch_dtype)
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
            ).to(self.device)

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
        
        if not self.offload:
            if is_contronet_enable:
                if self.first:
                    del self.pipeline
                    flush()
                
                controlnet_a = FluxControlNetModel.from_pretrained(local_path, torch_dtype=self.torch_dtype)
                controlnet=FluxMultiControlNetModel(controlnet_a)
                pipe = FluxControlNetPipeline.from_pretrained(self.models_dir, controlnet=controlnet, torch_dtype=torch.bfloat16)
                if is_lora_enable:
                    pipe.load_lora_weights(lora_local_path)
                    pipe.fuse_lora(lora_scale=lora_weight)
                
                control_image=load_image(controlnet_image)
                
            else:
                if self.first is not True:
                    FluxPipeline.from_pretrained(self.models_dir, torch_dtype=self.torch_dtype)
                else:
                    pipe = self.pipeline
                if is_lora_enable:
                    pipe.load_lora_weights(lora_local_path)
                    pipe.fuse_lora(lora_scale=lora_weight)
                pipe.enable_model_cpu_offload()
                image = pipe(prompt, 
                    num_inference_steps=24, 
                    guidance_scale=3.5,
                    width=768, height=1024,
                ).images
                
        else:
            if self.first is not True:
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
                ).to(self.device)
            with torch.no_grad():
                print("Encoding prompts...")
                gr.Info("编码提示词中...",duration=2)
                prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
                    prompt=prompt, prompt_2=None, max_sequence_length=256
                )
                
            del self.text_encoder
            del self.text_encoder_2
            del self.tokenizer
            del self.tokenizer_2
            del self.pipeline
              
            flush()
            self.first = False
                
            pipe = FluxPipeline.from_pretrained(
                self.models_dir,
                text_encoder=None,
                text_encoder_2=None,
                tokenizer=None,
                tokenizer_2=None,
                vae=None,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
                
            print("Running denoising...")
            gr.Info("开始降噪...")
            latents = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=num_steps,
                guidance_scale=true_gs,
                height=height,
                width=width,
                output_type="latent",
            ).images
            print(f"Latents Shape: {latents.shape=}")
                
            del pipe.transformer
            del pipe
                
            flush()
                
            vae = AutoencoderKL.from_pretrained(self.models_dir, revision="refs/pr/1", subfolder="vae", torch_dtype=self.torch_dtype).to(
                self.device
            )
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
            image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
            
            with torch.no_grad():
                print("Running decoding.")
                gr.Info("解码中...")
                
                latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
                latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

                images = vae.decode(latents, return_dict=False)[0]
                images = image_processor.postprocess(image, output_type="pil")
                images[0].save("image.png")        
                