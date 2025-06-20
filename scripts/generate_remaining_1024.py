#!/usr/bin/env python3
"""Generate remaining test images at 1024x1024 with 20 steps."""

import os
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

def main():
    """Generate remaining test images."""
    
    output_dir = Path("tests/data/images")
    
    print("Loading Stable Diffusion XL pipeline...")
    
    # Load SDXL pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    
    # Force CPU usage
    device = "cpu"
    pipe = pipe.to(device)
    
    print(f"Using device: {device}")
    
    # Enable memory efficient settings for CPU
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    
    # Remaining test cases (skip cat_office_typing.png which already exists)
    remaining_cases = [
        {
            "prompt": "photorealistic, hyper detailed, a golden retriever dog with paws on a computer keyboard in a modern office, professional lighting, sharp focus, 8k resolution, realistic fur texture, office environment with desk and monitor", 
            "filename": "dog_office_typing.png",
            "description": "Dog in office (indoor)"
        },
        {
            "prompt": "photorealistic, hyper detailed, a beautiful cat sitting on a rocky mountain peak during golden hour sunrise, majestic mountain landscape, natural lighting, sharp focus, 8k resolution, realistic fur texture, dramatic sky",
            "filename": "cat_mountain_sunrise.png", 
            "description": "Cat on mountain (outdoor)"
        },
        {
            "prompt": "photorealistic, hyper detailed, a german shepherd dog sitting on a rocky mountain peak during golden hour sunrise, majestic mountain landscape, natural lighting, sharp focus, 8k resolution, realistic fur texture, dramatic sky",
            "filename": "dog_mountain_sunrise.png",
            "description": "Dog on mountain (outdoor)"
        }
    ]
    
    # Generate remaining images
    for i, case in enumerate(remaining_cases, 2):  # Start from 2 since we have 1 already
        print(f"\nGenerating image {i}/4: {case['description']}")
        print(f"Prompt: {case['prompt']}")
        
        try:
            # Generate image with 20 diffusion steps at 1024x1024 resolution
            image = pipe(
                case["prompt"],
                num_inference_steps=20,  # Use 20 diffusion steps as requested
                guidance_scale=5.0,
                height=1024,
                width=1024
            ).images[0]
            
            # Save image
            output_path = output_dir / case["filename"]
            image.save(output_path)
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Error generating image {i}: {e}")
            continue
    
    print(f"\nCompleted generating remaining images in {output_dir}")

if __name__ == "__main__":
    main()