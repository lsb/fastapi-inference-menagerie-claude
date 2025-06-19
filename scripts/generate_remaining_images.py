#!/usr/bin/env python3
"""Generate remaining test images."""

import os
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline

def main():
    """Generate remaining 2 test images."""
    
    output_dir = Path("tests/data/images")
    
    print("Loading Stable Diffusion 1.5 pipeline...")
    
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Use MPS if available
    if torch.backends.mps.is_available():
        device = "mps"
        pipe = pipe.to(device)
    else:
        device = "cpu"
        pipe = pipe.to(device)
    
    print(f"Using device: {device}")
    
    # Enable memory efficient settings
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    
    # Remaining test cases
    remaining_cases = [
        {
            "prompt": "a cat sitting on a mountaintop at sunrise, beautiful landscape, detailed",
            "filename": "cat_mountain_sunrise.png", 
            "description": "Cat on mountain (outdoor)"
        },
        {
            "prompt": "a dog sitting on a mountaintop at sunrise, beautiful landscape, detailed",
            "filename": "dog_mountain_sunrise.png",
            "description": "Dog on mountain (outdoor)"
        }
    ]
    
    # Generate remaining images
    for i, case in enumerate(remaining_cases, 3):
        print(f"\nGenerating image {i}/4: {case['description']}")
        print(f"Prompt: {case['prompt']}")
        
        try:
            # Generate image
            image = pipe(
                case["prompt"],
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
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