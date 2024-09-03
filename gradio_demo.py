from pathlib import Path

import torch
import gradio as gr

from src.flux.xflux_pipeline import XFluxPipeline


def create_demo(
        model_type: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offload: bool = False,
        ckpt_dir: str = "",
    ):
    xflux_pipeline = XFluxPipeline(model_type, device, offload)
    checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))

    with gr.Blocks(title="X-Flux WebUI") as demo:
        gr.Markdown(f"# Flux 适配器 by XLabs AI - 模型: {model_type}")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="提示词（Prompt）", value="handsome woman in the city")

                with gr.Accordion("生成选项（Generation Options）", open=False):
                    with gr.Row():
                        width = gr.Slider(512, 2048, 1024, step=16, label="宽度（Width）")
                        height = gr.Slider(512, 2048, 1024, step=16, label="高度（Height）")
                    neg_prompt = gr.Textbox(label="负向提示词（Negative Prompt）", value="bad photo")
                    with gr.Row():
                        num_steps = gr.Slider(1, 50, 25, step=1, label="迭代步数（Number of steps）")
                        timestep_to_start_cfg = gr.Slider(1, 50, 1, step=1, label="timestep_to_start_cfg")
                    with gr.Row():
                        guidance = gr.Slider(1.0, 5.0, 4.0, step=0.1, label="引导系数（Guidance Scale）", interactive=True)
                        true_gs = gr.Slider(1.0, 5.0, 3.5, step=0.1, label="True Guidance Scale", interactive=True)
                    seed = gr.Textbox(-1, label="随机种子（Seed, -1 为随机")

                with gr.Accordion("ControlNet 选项（Options）", open=False):
                    control_type = gr.Dropdown(["canny", "hed", "depth"], label="Control 类型（Type）")
                    control_weight = gr.Slider(0.0, 1.0, 0.8, step=0.1, label="Controlnet 权重（Weight）", interactive=True)
                    local_path = gr.Dropdown(checkpoints, label="Controlnet 模型（Checkpoint）",
                        info="Controlnet 权重的本地路径 (如果没有, 将会从 Hugging Face 下载)"
                        )
                    controlnet_image = gr.Image(label="输入 Controlnet 图片", visible=True, interactive=True)

                with gr.Accordion("LoRA 选项（Options）", open=False):
                    lora_weight = gr.Slider(0.0, 1.0, 0.9, step=0.1, label="LoRA 权重（weight）", interactive=True)
                    lora_local_path = gr.Dropdown(
                        checkpoints, label="LoRA 模型（Checkpoint）", info="LoRA 模型的本地路径"
                        )

                with gr.Accordion("IP Adapter 选项（Options）", open=False):
                    image_prompt = gr.Image(label="图片提示（image_prompt）", visible=True, interactive=True)
                    ip_scale = gr.Slider(0.0, 1.0, 1.0, step=0.1, label="ip_scale")
                    neg_image_prompt = gr.Image(label="负向图片提示（neg_image_prompt）", visible=True, interactive=True)
                    neg_ip_scale = gr.Slider(0.0, 1.0, 1.0, step=0.1, label="neg_ip_scale")
                    ip_local_path = gr.Dropdown(
                        checkpoints, label="IP Adapter 模型（Checkpoint）",
                        info="IP Adapter 模型本地地址 (如果没有, 将会从 Hugging Face 下载)"
                        )
                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="生成的图片（Generated Image）")
                download_btn = gr.File(label="下载高分辨率图片（Download full-resolutiom）")

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
