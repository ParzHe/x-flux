import re
import os
import yaml
import tempfile
import subprocess
from pathlib import Path

import torch
import gradio as gr

from src.flux.xflux_pipeline import XFluxPipeline
from src.flux.diffusers_pipeline import DiffusersFluxPipeline

import time

from src.utils import get_gpu_mem_info, flush

def list_dirs(path):
    if path is None or path == "None" or path == "":
        return

    if not os.path.exists(path):
        path = os.path.dirname(path)
        if not os.path.exists(path):
            return

    if not os.path.isdir(path):
        path = os.path.dirname(path)

    def natural_sort_key(s, regex=re.compile("([0-9]+)")):
        return [
            int(text) if text.isdigit() else text.lower() for text in regex.split(s)
        ]

    subdirs = [
        (item, os.path.join(path, item))
        for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item))
    ]
    subdirs = [
        filename
        for item, filename in subdirs
        if item[0] != "." and item not in ["__pycache__"]
    ]
    subdirs = sorted(subdirs, key=natural_sort_key)
    if os.path.dirname(path) != "":
        dirs = [os.path.dirname(path), path] + subdirs
    else:
        dirs = [path] + subdirs

    if os.sep == "\\":
        dirs = [d.replace("\\", "/") for d in dirs]
    for d in dirs:
        yield d

def list_train_data_dirs():
    current_train_data_dir = "."
    return list(list_dirs(current_train_data_dir))

def update_config(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_config(d.get(k, {}), v)
        else:
            # convert Gradio components to strings
            if hasattr(v, 'value'):
                d[k] = str(v.value)
            else:
                try:
                    d[k] = int(v)
                except (TypeError, ValueError):
                    d[k] = str(v)
    return d

def start_lora_training(
        data_dir: str, output_dir: str, lr: float, steps: int, rank: int
    ):
    inputs = {
        "data_config": {
            "img_dir": data_dir,
            },
            "output_dir": output_dir,
            "learning_rate": lr,
            "rank": rank,
            "max_train_steps": steps,
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creating folder {output_dir} for the output checkpoint file...")
        gr.Info(f"创建 {output_dir} 文件夹用于放置模型文件...")

    script_path = Path(__file__).resolve()
    config_path = script_path.parent / "train_configs" / "casdao_lora.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config = update_config(config, inputs)
    print("Config file is updated...", config)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as temp_file:
        yaml.dump(config, temp_file, default_flow_style=False)
        tmp_config_path = temp_file.name

    command = ["accelerate", "launch", "train_flux_lora_deepspeed.py", "--config", tmp_config_path]
    result = subprocess.run(command, check=True)

    # rRemove the temporary file after the command is run
    Path(tmp_config_path).unlink()

    return result

def init_pipeline(pipeline_type: str = "xflux", model_type: str = "flux-dev", device: str | torch.device = "cuda", offload: bool = False):
    print("初始化Pipeline中...")
    flush()
    if pipeline_type == "xflux":
        pipeline=XFluxPipeline(model_type, device, offload)
    else:
        pipeline=DiffusersFluxPipeline(model_type,device,offload)
    print("初始化Pipeline完成")
    steps=0
    guidance=0
    if model_type=="flux-schnell":
        steps=4
        guidance=0.0
    else:
        steps=40
        guidance=3.5
    return pipeline, steps, guidance

class casdao_xflux_ui:
    def __init__(self, pipeline_type: str, model_type: str, device: str, offload: bool = False, 
                 ckpt_dir: str="", output_path: str = ""):
        
        self.pipeline_type=pipeline_type
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type
        self.ckpt_dir=ckpt_dir
        self.output_path=output_path
        
        self.gpu_mem_total, self.gpu_mem_used, self.gpu_mem_free = get_gpu_mem_info(gpu_id=0)
        
        self.pipeline_list=["xflux","diffusers"]
        if self.gpu_mem_total> 25:
            self.model_list=["flux-dev","flux-schnell"]
        else:
            self.model_list=["flux-dev","flux-dev-fp8","flux-schnell"]
        
        self.pipeline, self.init_steps,self.init_gs=init_pipeline(pipeline_type,model_type,device,offload)
        self.controlnet_checkpoints=sorted(Path(self.ckpt_dir+"/Controlnet").glob("*.safetensors"))
        self.lora_checkpoints=sorted(Path(self.ckpt_dir+"/LoRA").glob("*.safetensors"))
        self.ip_checkpoints=sorted(Path(self.ckpt_dir+"/IP_Adapter").glob("*.safetensors"))
        
        self.css="""
            nav {
                text-align: center;
            }
            .shield {
                margin-right:5px;
            }
            .shields {
                align:center;
                margin:auto;
                display:flex;
            }
            .enable_button {
                align:center;
            }
            #generate_btn {
                margin-left: 20px;
                color: orange;
            }
            """

    
    def create_demo(self):
        with gr.Blocks(title="X-Flux-WebUI",css=self.css) as demo:
            gr.Markdown(f"# X-Flux-WebUI：由 XLabs AI 推出的 Flux Adapter")
            gr.HTML(
                """
                <div>
                    <div class="shields">
                        <div class="shield">
                            <a href="https://ai.casdao.com/">
                                <img src="https://img.shields.io/badge/Casdao-%E6%99%BA%E7%AE%97%E7%A9%BA%E9%97%B4-blue" alt="算力互联-智算空间" height="50">
                            </a>
                        </div>
                        <div class="shield">
                            <a href="https://github.com/xlabs-ai">
                                <img src="https://img.shields.io/badge/XLabs-AI-green" alt="XLabs-AI" height="50">
                            </a>
                        </div>
                        <div class="shield">
                            <a href="https://github.com/XLabs-AI/x-flux">
                                <img src="https://img.shields.io/github/stars/XLabs-AI/x-flux?style=social" alt="GitHub标星" height="50">
                            </a>
                        </div>
                        <!-- div class="shield">
                            <a href="https://blackforestlabs.ai/">
                                <img alt="Black Forest Labs" src="https://img.shields.io/badge/Black_Forest_Labs-gray" height="50">
                            </a>
                        </div -->
                    </div>
                </div>
                """
            )
            with gr.Row():
                pipeline_dropdown=gr.Dropdown(label="推理管线（Pipeline）",choices=self.pipeline_list,value=self.pipeline_type)
                model_checkpoint=gr.Dropdown(label="模型（Checkpoint）",choices=self.model_list,value=self.model_type,scale=5, interactive=True if self.gpu_mem_total>25 else False)
                device_dropdown=gr.Dropdown(label="设备（Device）",choices=["cpu","cuda"],value=self.device,visible=False, scale=0,allow_custom_value=True)
                offload_checkbox=gr.Checkbox(label="低内存模式（Offload for Low VRAM）",
                                            # info="4090及以下的显卡不使用FP8模型时一定要勾选！",
                                            value=self.offload,
                                            scale=1,
                                            container=True,
                                            interactive= False,
                                            visible = False if self.gpu_mem_total>25 else True)
                
            with gr.Tab("推理（Inference）"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row(elem_classes="enable_button"):
                            is_contronet_enable=gr.Checkbox(label="启用ControlNet",container=True,elem_classes="enable_button")
                            is_lora_enable=gr.Checkbox(label="启用LoRA",container=True,elem_classes="enable_button")
                            is_ip_enable=gr.Checkbox(label="启用IP Adpater",container=True,elem_classes="enable_button",visible=True if self.pipeline_type=="xflux" else False)
                            generate_btn = gr.Button("生成（Generate）",elem_id="generate_btn")
                        with gr.Accordion(label="提示词（Prompt）",open=True):
                            with gr.Row():
                                prompt = gr.Textbox(
                                    label="正面提示词（Positive Prompt）", 
                                    placeholder="使用英文输入正文提示词，即提示希望模型生成的内容",
                                    value="a handsome asian woman in the city",
                                    container=True)
                            with gr.Row():
                                neg_prompt = gr.Textbox(
                                    label="负面提示词（Negative Prompt）", 
                                    # info="负面提示词及提示模型不要生成的内容，如bad photo。需要输入英文",
                                    placeholder="使用英文，输入负面提示词，即不希望模型生成的内容",
                                    value="bad photo",
                                    container=True,
                                    visible=True if self.pipeline_type=="xflux" else False
                                )
                            
                        with gr.Accordion("生成设置（Generation Options）", open=True):
                            with gr.Row():
                                width = gr.Slider(512, 2048, 1024, step=16, label="宽度（Width）")
                                height = gr.Slider(512, 2048, 1024, step=16, label="高度（Height）")
                            
                            with gr.Row():
                                num_steps = gr.Slider(1, 100, self.init_steps, step=1, label="迭代步数（Number of steps）")
                                timestep_to_start_cfg = gr.Slider(1, 50, 1, step=1, label="timestep_to_start_cfg",visible=True if self.pipeline_type=="xflux" else False)
                            
                            with gr.Row():
                                guidance = gr.Slider(0.0, 10.0, 4.0 if not self.init_gs==0 else 0, step=0.1, label="引导（Guidance）", interactive=True, visible=True if self.pipeline_type=="xflux" else False)
                                true_gs = gr.Slider(0.0, 10.0, self.init_gs, step=0.1, label="True Guidance", interactive=True, )
                            
                            seed = gr.Textbox(-1, label="随机种子（Seed，-1 为随机）")
                        
                        with gr.Accordion("ControlNet 设置（需启用 ControlNet 才有效）", open=False, elem_id="controlnet_options"):
                            # is_contronet_enable=gr.Checkbox(label="启用（Enable）",container=True,scale=1)
                            with gr.Row():
                                control_type = gr.Dropdown(["canny", "hed", "depth"], value="canny",label="Control 类型（type）",scale=1)
                                local_path = gr.Dropdown(self.controlnet_checkpoints, 
                                    value=self.controlnet_checkpoints[0],
                                    label="Controlnet 模型（Checkpoint）",
                                    info="Controlnet 模型的本地地址（如果无, 将会从 Hugging Face 下载。）",
                                    scale=2
                                )
                            control_weight = gr.Slider(0.0, 1.0 if self.pipeline_type=="xflux" else 5.0, 0.8, step=0.1, label="Controlnet 权重（weight）", interactive=True)
                            controlnet_image = gr.Image(label="输入的 Controlnet 图片", visible=True, interactive=True)
                        
                        with gr.Accordion("LoRA 设置（需启用 LoRA 才有效）", open=False, elem_id="lora_options"):
                            # is_lora_enable=gr.Checkbox(label="启用（Enable）",container=True,scale=1)
                            with gr.Row():
                                lora_local_path = gr.Dropdown(
                                    self.lora_checkpoints, value=self.lora_checkpoints[0],
                                    label="LoRA 模型（Checkpoint）", 
                                    # info="LoRA 模型本地地址",
                                    scale=3
                                )
                                lora_weight = gr.Slider(0.0, 1.0 if self.pipeline_type=="xflux" else 3.0, 0.9, step=0.1, label="LoRA 权重（Weight）", interactive=True,scale=3)
                        
                        with gr.Accordion("IP Adapter 设置（需启用 IP Adaptet 才有效）", open=False, visible=True if self.pipeline_type=="xflux" else False, elem_id="ip_options"):
                            # is_ip_enable=gr.Checkbox(label="启用（Enable）",container=True)
                            with gr.Accordion("正面图片提示设置（Positive Image Prompt Options）",open=True):
                                image_prompt = gr.Image(label="image_prompt", visible=True, interactive=True)
                                ip_scale = gr.Slider(0.0, 1.0, 1.0, step=0.1, label="ip_scale")
                            with gr.Accordion("负面图片提示设置（Negative Image Prompt Options）",open=False):
                                neg_image_prompt = gr.Image(label="neg_image_prompt", visible=True, interactive=True)
                                neg_ip_scale = gr.Slider(0.0, 1.0, 1.0, step=0.1, label="neg_ip_scale")
                            ip_local_path = gr.Dropdown(
                                self.ip_checkpoints, 
                                value=self.ip_checkpoints[0],
                                label="IP Adapter 模型（Checkpoint）",
                                info="IP Adapter 模型的本地地址（如果没有，将会从Hugging Face)",
                                visible=False,
                            )   
                        
                        # generate_btn = gr.Button("生成（Generate）")

                    with gr.Column():
                        output_dir = gr.Textbox(label="图片生成地址（Local path of generated image）", value=self.output_path,visible=False)
                        output_image = gr.Image(label="生成的图片（Generated Image）")
                        download_btn = gr.File(label="下载高清图片（Download full-resolution）")
                        max_vram = gr.Textbox(label="生成时峰值显存占用（Maximum Useed VRAM）",value=f"无生成，无数据")
                
                def update_pipeline(pipeline_type, model_type, device, offload):
                    gr.Info("切换Flux管线中...",duration=5)
                    del self.pipeline
                    flush()
                    if pipeline_type=="xflux":
                        self.pipeline=XFluxPipeline(model_type, device, offload)
                        enable_xflux_funcitons=True
                    else:
                        self.pipeline=DiffusersFluxPipeline(model_type,device,offload)
                        enable_xflux_funcitons=False
                    gr.Info("切换Flux管线完成！",duration=2)
                    outputs=[
                        pipeline_type,
                        gr.update(visible=enable_xflux_funcitons),# is_ip_enable
                        gr.update(visible=enable_xflux_funcitons), # neagetive prompt
                        gr.update(visible=enable_xflux_funcitons), # timesteps
                        gr.update(visible=enable_xflux_funcitons),# guidance
                        gr.update(maximum=1.0 if self.pipeline_type=="xflux" else 5.0), # control_weight
                        gr.update(maximum=1.0 if self.pipeline_type=="xflux" else 3.0) # LoRA weight
                    ]
                    return outputs

                def update_model(pipeline_type, model_type, device, offload):
                    gr.Info("切换Flux模型中...",duration=5)
                    del self.pipeline
                    flush()
                    if pipeline_type=="xflux":
                        self.pipeline=XFluxPipeline(model_type, device, offload)
                    else:
                        self.pipeline=DiffusersFluxPipeline(model_type,device,offload)
                    gr.Info("切换Flux模型完成！",duration=2)
                    
                    if model_type == "flux-schnell":
                        steps=4
                        guidance=0
                    else:
                        steps=28
                        guidance=3.5
                    
                    if self.gpu_mem_total < 35:
                        return model_type, device, True, steps, guidance
                    else: 
                        return model_type, device, offload, steps, guidance
                
                def generate(prompt, image_prompt, controlnet_image, width, height, guidance,
                        num_steps, seed, true_gs, 
                        is_ip_enable, ip_scale, neg_ip_scale, neg_prompt,
                        neg_image_prompt, timestep_to_start_cfg, 
                        is_contronet_enable, control_type, control_weight,
                        is_lora_enable, lora_weight, 
                        local_path, lora_local_path, ip_local_path, output_dir,
                    ):
                    
                    gr.Info("开始生成...",duration=5)
                    print("开始生成...")
                    flush()
                    start_time = time.time()
                    
                    img,filename = self.pipeline.gradio_generate(prompt, image_prompt, 
                        controlnet_image, width, height, guidance,
                        num_steps, seed, true_gs, 
                        is_ip_enable, ip_scale, neg_ip_scale, neg_prompt,
                        neg_image_prompt, timestep_to_start_cfg, is_contronet_enable, control_type, control_weight,
                        is_lora_enable, lora_weight, 
                        local_path, lora_local_path, ip_local_path, output_dir)
                    
                    elapsed_time = time.time()-start_time
                    max_vram_used = torch.cuda.max_memory_allocated() / 1024 / 1024 /1024
                    gr.Info("生成完毕",duration=2)
                    print("生成完毕")
                    print(f"生成耗费的时间：{elapsed_time:2f} 秒")
                    print(f"峰值显存占用: {max_vram_used:2f} GB")
                    torch.cuda.empty_cache()
                    max_vram_used=f"{max_vram_used:2f} GB"
                    return img,filename,max_vram_used,"生成（Generate）"
                
                pipeline_dropdown.change(
                    fn=update_pipeline,
                    inputs=[pipeline_dropdown,model_checkpoint,device_dropdown,offload_checkbox],
                    outputs=[pipeline_dropdown,is_ip_enable,neg_prompt,timestep_to_start_cfg,guidance,control_weight,lora_weight]
                )
                
                gr.on(
                    triggers=[model_checkpoint.change,offload_checkbox.change],
                    fn=update_model,
                    inputs=[pipeline_dropdown,model_checkpoint,device_dropdown,offload_checkbox],
                    outputs=[model_checkpoint,device_dropdown,offload_checkbox,num_steps,true_gs],
                )
                
                inputs = [
                        prompt, image_prompt, controlnet_image, width, height, guidance,
                        num_steps, seed, true_gs, 
                        is_ip_enable,ip_scale, neg_ip_scale, neg_prompt,
                        neg_image_prompt, timestep_to_start_cfg, 
                        is_contronet_enable, control_type, control_weight,
                        is_lora_enable, lora_weight, 
                        local_path, lora_local_path, ip_local_path, output_dir
                ]
                
                generate_btn.click(
                    fn=generate,
                    scroll_to_output=True,
                    inputs=inputs,
                    outputs=[output_image, download_btn,max_vram,generate_btn],
                )

            with gr.Tab("LoRA Finetuning",visible=False):
                data_dir =  gr.Dropdown(list_train_data_dirs(),
                                        label="训练图片 (directory containing the training images)",
                                        info="包含训练图片的文件夹。",
                                        )
                lora_output_dir = gr.Textbox(label="Output Path", value="lora_checkpoint")

                with gr.Accordion("训练设置（Training Options）", open=True):
                    lr = gr.Textbox(label="学习率（Learning Rate）", value="1e-5")
                    steps = gr.Slider(10000, 20000, 20000, step=100, label="训练步数（Train Steps）")
                    rank = gr.Slider(1, 100, 16, step=1, label="LoRA Rank")

                training_btn = gr.Button("开始训练（Start Traininng）")
                training_btn.click(
                    fn=start_lora_training,
                    inputs=[data_dir, lora_output_dir, lr, steps, rank],
                    outputs=[],
                )
                
        return demo
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="flux-dev", help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--port",type=int, default=7860,help="The server port of the gradio demo")
    parser.add_argument("--ckpt_dir", type=str, default=".", help="Folder with checkpoints in safetensors format")
    parser.add_argument("--output_dir", type=str, default="./output/gradio", help="Folder of output frome inference")
    parser.add_argument("--pipeline",type=str,default="xflux",help="Pipeline of Flux in inference")
    args = parser.parse_args()

    ui = casdao_xflux_ui(
        pipeline_type=args.pipeline,
        model_type=args.name,
        device=args.device,
        offload=args.offload,
        ckpt_dir=args.ckpt_dir, 
        output_path=args.output_dir,)
    ui.create_demo().launch(share=args.share,server_name="0.0.0.0",server_port=args.port)
