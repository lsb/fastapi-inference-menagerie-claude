"""Qwen 2.5 VL adapter for visual question answering."""

import logging
from typing import Dict, Any, AsyncGenerator
import asyncio

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image

from services.common.adapter import ModelAdapter
from services.common.utils import decode_base64_to_image

logger = logging.getLogger(__name__)


class QwenVLAdapter(ModelAdapter):
    """Qwen 2.5 VL adapter for visual question answering."""
    
    def __init__(self, gcs_path: str, device: str) -> None:
        """Initialize Qwen VL adapter."""
        super().__init__(gcs_path, device)
        self.model = None
        self.tokenizer = None
        self.processor = None
    
    async def load_model(self) -> None:
        """Load Qwen 2.5 VL model."""
        logger.info("Loading Qwen 2.5 VL model...")
        
        try:
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
            
            # Load tokenizer and processor
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            # Load model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
                device_map=self.device if self.device.startswith('cuda') else None
            )
            
            if not self.device.startswith('cuda'):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            self._loaded = True
            logger.info(f"Qwen 2.5 VL model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen VL model: {e}")
            raise
    
    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run visual question answering.
        
        Args:
            payload: Input containing:
                - image: base64 encoded image
                - question: question about the image
                - max_tokens: maximum tokens to generate (default: 256)
                - temperature: sampling temperature (default: 0.1)
                
        Returns:
            Dictionary with generated answer
        """
        self._ensure_loaded()
        
        image_b64 = payload['image']
        question = payload['question']
        max_tokens = payload.get('max_tokens', 256)
        temperature = payload.get('temperature', 0.1)
        
        # Decode image
        image = decode_base64_to_image(image_b64)
        
        # Prepare conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Apply chat template and process inputs
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        answer = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return {
            "answer": answer,
            "question": question,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    
    async def stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream visual question answering response.
        
        Args:
            payload: Same as predict method
            
        Yields:
            Streaming response tokens
        """
        self._ensure_loaded()
        
        image_b64 = payload['image']
        question = payload['question']
        max_tokens = payload.get('max_tokens', 256)
        temperature = payload.get('temperature', 0.1)
        
        # Decode image
        image = decode_base64_to_image(image_b64)
        
        # Prepare conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Apply chat template and process inputs
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Streaming generation
        with torch.no_grad():
            input_length = inputs.input_ids.shape[1]
            
            for _ in range(max_tokens):
                # Generate next token
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                
                if temperature > 0:
                    # Sample with temperature
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Decode token
                token_text = self.tokenizer.decode(next_token, skip_special_tokens=True)
                
                # Yield token
                yield token_text
                
                # Update inputs for next iteration
                inputs.input_ids = torch.cat([inputs.input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Add small delay to avoid overwhelming the client
                await asyncio.sleep(0.01)