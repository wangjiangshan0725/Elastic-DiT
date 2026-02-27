import torch
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image_elastic_dit import (EDiTQwenImagePipeline,
                                                        ModelConfig)

model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ]

model_configs[0].path = [
            "ckpt/Qwen-Image/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
            "ckpt/Qwen-Image/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
            "ckpt/Qwen-Image/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
            "ckpt/Qwen-Image/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
            "ckpt/Qwen-Image/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
            "ckpt/Qwen-Image/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
            "ckpt/Qwen-Image/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
            "ckpt/Qwen-Image/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
            "ckpt/Qwen-Image/transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
        ]
        
model_configs[1].path = [
    "ckpt/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "ckpt/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "ckpt/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "ckpt/Qwen-Image/text_encoder/model-00004-of-00004.safetensors",
]
model_configs[2].path = [
    "ckpt/Qwen-Image/vae/diffusion_pytorch_model.safetensors",
]

pipe = EDiTQwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=model_configs, 
    tokenizer_config=ModelConfig('ckpt/Qwen-Image/tokenizer'),
)
state_dict = load_state_dict("ckpt/model.safetensors")

pipe.dit.load_state_dict(state_dict)
prompt = 'A man in a suit is standing in front of the window, looking at the bright moon outside the window. The man is holding a yellowed paper with handwritten words on it: "A lantern moon climbs through the silver night, Unfurling quiet dreams across the sky, Each star a whispered promise wrapped in light, That dawn will bloom, though darkness wanders by." There is a cute cat on the windowsill'

image = pipe(prompt, seed=0, num_inference_steps=30)
image.save("results.jpg")
