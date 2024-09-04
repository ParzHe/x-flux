import re
import os
import yaml
import tempfile
import subprocess
from pathlib import Path

import torch
import gradio as gr

from src.flux.xflux_pipeline import XFluxPipeline


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


def create_demo(
        model_type: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offload: bool = False,
        ckpt_dir: str = "",
    ):
    xflux_pipeline = XFluxPipeline(model_type, device, offload)
    checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))

    with gr.Blocks(title="X-Flux") as demo:
        gr.Markdown(f"# Flux 适配器 by XLabs AI - Model: {model_type}")
        with gr.Tab("推理（Inference）"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="提示词（Prompt）", value="handsome woman in the city")

                    with gr.Accordion("生成设置（Generation Options）", open=False):
                        with gr.Row():
                            width = gr.Slider(512, 2048, 1024, step=16, label="宽度（Width）")
                            height = gr.Slider(512, 2048, 1024, step=16, label="高度（Height）")
                        neg_prompt = gr.Textbox(label="负面提示词（Negative Prompt）", value="bad photo")
                        with gr.Row():
                            num_steps = gr.Slider(1, 50, 25, step=1, label="迭代步数（Number of steps）")
                            timestep_to_start_cfg = gr.Slider(1, 50, 1, step=1, label="timestep_to_start_cfg")
                        with gr.Row():
                            guidance = gr.Slider(1.0, 5.0, 4.0, step=0.1, label="引导（Guidance）", interactive=True)
                            true_gs = gr.Slider(1.0, 5.0, 3.5, step=0.1, label="True Guidance", interactive=True)
                        seed = gr.Textbox(-1, label="随机种子（Seed，-1 为随机）")

                    with gr.Accordion("ControlNet 设置（Options）", open=False):
                        control_type = gr.Dropdown(["canny", "hed", "depth"], label="Control 类型（type）")
                        control_weight = gr.Slider(0.0, 1.0, 0.8, step=0.1, label="Controlnet 权重（weight）", interactive=True)
                        local_path = gr.Dropdown(checkpoints, label="Controlnet 模型（Checkpoint）",
                            info="Controlnet 模型的本地地址（如果无, 将会从 Hugging Face 下载。）"
                            )
                        controlnet_image = gr.Image(label="输入的 Controlnet 图片", visible=True, interactive=True)

                    with gr.Accordion("LoRA 设置（Options）", open=False):
                        lora_weight = gr.Slider(0.0, 1.0, 0.9, step=0.1, label="LoRA 权重（Weight）", interactive=True)
                        lora_local_path = gr.Dropdown(
                            checkpoints, label="LoRA 模型（Checkpoint）", info="LoRA 模型本地地址"
                            )

                    with gr.Accordion("IP Adapter 设置（Options）", open=False):
                        image_prompt = gr.Image(label="image_prompt", visible=True, interactive=True)
                        ip_scale = gr.Slider(0.0, 1.0, 1.0, step=0.1, label="ip_scale")
                        neg_image_prompt = gr.Image(label="neg_image_prompt", visible=True, interactive=True)
                        neg_ip_scale = gr.Slider(0.0, 1.0, 1.0, step=0.1, label="neg_ip_scale")
                        ip_local_path = gr.Dropdown(
                            checkpoints, label="IP Adapter 模型（Checkpoint）",
                            info="IP Adapter 模型的本地地址（如果没有，将会从Hugging Face)"
                            )
                    generate_btn = gr.Button("生成（Generate）")

                with gr.Column():
                    output_image = gr.Image(label="生成的图片（Generated Image）")
                    download_btn = gr.File(label="下载高清图片（Download full-resolution）")

            inputs = [prompt, image_prompt, controlnet_image, width, height, guidance,
                    num_steps, seed, true_gs, ip_scale, neg_ip_scale, neg_prompt,
                    neg_image_prompt, timestep_to_start_cfg, control_type, control_weight,
                    lora_weight, local_path, lora_local_path, ip_local_path
                    ]
            generate_btn.click(
                fn=xflux_pipeline.gradio_generate,
                inputs=inputs,
                outputs=[output_image, download_btn],
            )

        with gr.Tab("LoRA Finetuning"):
            data_dir =  gr.Dropdown(list_train_data_dirs(),
                                    label="训练图片 (directory containing the training images)",
                                    info="包含训练图片的文件夹。",
                                    )
            output_dir = gr.Textbox(label="Output Path", value="lora_checkpoint")

            with gr.Accordion("训练设置（Training Options）", open=True):
                lr = gr.Textbox(label="学习率（Learning Rate）", value="1e-5")
                steps = gr.Slider(10000, 20000, 20000, step=100, label="训练步数（Train Steps）")
                rank = gr.Slider(1, 100, 16, step=1, label="LoRA Rank")

            training_btn = gr.Button("开始训练（Start Traininng）")
            training_btn.click(
                fn=start_lora_training,
                inputs=[data_dir, output_dir, lr, steps, rank],
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
    args = parser.parse_args()

    demo = create_demo(args.name, args.device, args.offload, args.ckpt_dir)
    demo.launch(share=args.share,server_name="0.0.0.0",server_port=args.port)
