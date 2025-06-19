#!/usr/bin/env python3
"""
CLIP Usage Examples

This script demonstrates various ways to use the CLIP model service
for text and image encoding, similarity computation, and more.
"""

import base64
import requests
from pathlib import Path
from typing import List, Dict, Any


class CLIPClient:
    """Client for interacting with CLIP model service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize CLIP client.
        
        Args:
            base_url: Base URL of the CLIP service
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def encode_texts(self, texts: List[str]) -> Dict[str, Any]:
        """Encode text inputs to embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        response = self.session.post(
            f"{self.base_url}/v1/clip/encode/text",
            json={"texts": texts}
        )
        response.raise_for_status()
        return response.json()["result"]
    
    def encode_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """Encode images to embeddings.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        # Convert images to base64
        images_b64 = []
        for path in image_paths:
            with open(path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
                images_b64.append(image_b64)
        
        response = self.session.post(
            f"{self.base_url}/v1/clip/encode/image",
            json={"images": images_b64}
        )
        response.raise_for_status()
        return response.json()["result"]
    
    def compute_similarity(self, texts: List[str], image_paths: List[str]) -> Dict[str, Any]:
        """Compute text-image similarity matrix.
        
        Args:
            texts: List of text descriptions
            image_paths: List of paths to image files
            
        Returns:
            Dictionary containing similarity matrix and metadata
        """
        # Convert images to base64
        images_b64 = []
        for path in image_paths:
            with open(path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
                images_b64.append(image_b64)
        
        response = self.session.post(
            f"{self.base_url}/v1/clip/similarity",
            json={
                "texts": texts,
                "images": images_b64
            }
        )
        response.raise_for_status()
        return response.json()["result"]
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


def example_text_encoding():
    """Example: Encode text descriptions."""
    print("🔤 Text Encoding Example")
    print("=" * 40)
    
    client = CLIPClient()
    
    # Example texts
    texts = [
        "a cat sitting on a mat",
        "a dog playing in the park",
        "a bird flying in the sky",
        "a fish swimming in water",
        "a beautiful sunset over mountains"
    ]
    
    # Encode texts
    result = client.encode_texts(texts)
    
    print(f"Encoded {result['count']} texts")
    print(f"Embedding shape: {result['shape']}")
    print(f"First embedding (first 5 dims): {result['embeddings'][0][:5]}")
    print()


def example_image_encoding():
    """Example: Encode images (requires image files)."""
    print("🖼️ Image Encoding Example")
    print("=" * 40)
    
    # Note: This example assumes you have image files
    # For demo purposes, we'll show the structure
    
    print("This example requires actual image files.")
    print("Usage:")
    print("  client = CLIPClient()")
    print("  result = client.encode_images(['cat.jpg', 'dog.jpg'])")
    print("  print(f'Encoded {result[\"count\"]} images')")
    print()


def example_similarity_search():
    """Example: Find best matching images for text queries."""
    print("🔍 Similarity Search Example")
    print("=" * 40)
    
    # Note: This example shows the structure for similarity computation
    print("This example shows how to find the best matching images for text queries.")
    print()
    print("Usage:")
    print("  texts = ['a cute cat', 'a playful dog']")
    print("  images = ['image1.jpg', 'image2.jpg', 'image3.jpg']")
    print("  result = client.compute_similarity(texts, images)")
    print("  ")
    print("  # Find best match for each text")
    print("  for i, text in enumerate(texts):")
    print("      similarities = result['similarity_matrix'][i]")
    print("      best_match_idx = similarities.index(max(similarities))")
    print("      print(f'Best match for \"{text}\": {images[best_match_idx]}')")
    print()


def example_batch_processing():
    """Example: Process multiple queries efficiently."""
    print("⚡ Batch Processing Example")
    print("=" * 40)
    
    client = CLIPClient()
    
    # Large batch of texts
    texts = [
        f"example text number {i}" for i in range(50)
    ]
    
    print(f"Processing batch of {len(texts)} texts...")
    
    try:
        result = client.encode_texts(texts)
        print(f"✅ Successfully processed {result['count']} texts")
        print(f"Embedding dimensions: {result['shape'][1]}")
    except Exception as e:
        print(f"❌ Error processing batch: {e}")
    
    print()


def example_health_monitoring():
    """Example: Monitor service health."""
    print("🏥 Health Monitoring Example")
    print("=" * 40)
    
    client = CLIPClient()
    
    try:
        health = client.health_check()
        print(f"Service status: {health['status']}")
        print(f"Model class: {health['model_class']}")
        print(f"Device: {health['device']}")
        print(f"GCS path: {health['gcs_path']}")
    except Exception as e:
        print(f"❌ Service health check failed: {e}")
    
    print()


def example_error_handling():
    """Example: Proper error handling."""
    print("⚠️ Error Handling Example")
    print("=" * 40)
    
    client = CLIPClient("http://localhost:9999")  # Wrong port
    
    try:
        result = client.encode_texts(["test"])
        print("✅ Request successful")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - service may be down")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print()


def main():
    """Run all examples."""
    print("🦁 CLIP Service Usage Examples")
    print("=" * 50)
    print()
    
    # Check if service is available
    try:
        client = CLIPClient()
        health = client.health_check()
        print(f"✅ CLIP service is running (status: {health['status']})")
        print()
    except Exception:
        print("❌ CLIP service is not available")
        print("Start the service with: ./scripts/dev/run_local.sh clip")
        print("Or deploy to k3d: zoo deploy clip")
        print()
    
    # Run examples
    example_text_encoding()
    example_image_encoding()
    example_similarity_search()
    example_batch_processing()
    example_health_monitoring()
    example_error_handling()
    
    print("🎉 Examples completed!")
    print()
    print("Next steps:")
    print("- Try with your own images and texts")
    print("- Check the API docs at http://localhost:8000/docs")
    print("- Monitor metrics at http://localhost:8000/metrics")


if __name__ == "__main__":
    main()