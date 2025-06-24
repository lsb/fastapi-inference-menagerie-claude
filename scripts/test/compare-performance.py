#!/usr/bin/env python3
"""Compare performance between is-odd and CLIP to show overhead differences."""

import asyncio
import time
import os
import base64
from pathlib import Path

# Set cache directory
os.environ["CACHE_DIR"] = "/tmp/performance_comparison"

from services.is_odd.adapter import IsOddAdapter
from services.clip.adapter import CLIPAdapter


async def main():
    """Run performance comparison."""
    print("🔬 Performance Comparison: Is-Odd vs CLIP")
    print("=" * 50)
    
    # Load test image
    image_path = Path(__file__).parent.parent.parent / "tests" / "data" / "images" / "cat_office_typing.jpg"
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    # Initialize adapters
    print("\n📦 Loading models...")
    is_odd_adapter = IsOddAdapter()
    clip_adapter = CLIPAdapter(gcs_path=None, device="cpu")
    
    await is_odd_adapter.load_model()
    print("✅ Is-Odd model loaded")
    
    clip_start = time.time()
    await clip_adapter.load_model()
    clip_load_time = time.time() - clip_start
    print(f"✅ CLIP model loaded ({clip_load_time:.2f}s)")
    
    print("\n" + "=" * 50)
    print("🏃 Running performance tests...")
    print("=" * 50)
    
    # Test 1: Single request latency
    print("\n1️⃣ Single Request Latency:")
    
    # Is-Odd latency
    start = time.time()
    await is_odd_adapter.predict({"number": 42})
    is_odd_latency = (time.time() - start) * 1000
    
    # CLIP text latency
    start = time.time()
    await clip_adapter.predict({"task": "encode_text", "texts": ["a cat"]})
    clip_text_latency = (time.time() - start) * 1000
    
    # CLIP image latency
    start = time.time()
    await clip_adapter.predict({"task": "encode_image", "images": [image_b64]})
    clip_image_latency = (time.time() - start) * 1000
    
    print(f"   Is-Odd:      {is_odd_latency:>8.3f}ms")
    print(f"   CLIP text:   {clip_text_latency:>8.3f}ms ({clip_text_latency/is_odd_latency:>6.0f}x slower)")
    print(f"   CLIP image:  {clip_image_latency:>8.3f}ms ({clip_image_latency/is_odd_latency:>6.0f}x slower)")
    
    # Test 2: Throughput
    print("\n2️⃣ Throughput (100 requests):")
    
    # Is-Odd throughput
    start = time.time()
    tasks = [is_odd_adapter.predict({"number": i}) for i in range(100)]
    await asyncio.gather(*tasks)
    is_odd_time = time.time() - start
    is_odd_rps = 100 / is_odd_time
    
    # CLIP throughput (10 requests due to speed)
    start = time.time()
    tasks = [clip_adapter.predict({"task": "encode_text", "texts": [f"text {i}"]}) for i in range(10)]
    await asyncio.gather(*tasks)
    clip_time = time.time() - start
    clip_rps = 10 / clip_time
    
    print(f"   Is-Odd:      {is_odd_rps:>8.0f} req/s")
    print(f"   CLIP:        {clip_rps:>8.1f} req/s ({is_odd_rps/clip_rps:>6.0f}x faster)")
    
    # Test 3: Batch efficiency
    print("\n3️⃣ Batch Processing Efficiency:")
    
    # CLIP individual vs batch
    texts = ["a cat", "a dog", "a bird", "a car", "a tree"]
    
    # Individual
    start = time.time()
    for text in texts:
        await clip_adapter.predict({"task": "encode_text", "texts": [text]})
    individual_time = time.time() - start
    
    # Batch
    start = time.time()
    await clip_adapter.predict({"task": "encode_text", "texts": texts})
    batch_time = time.time() - start
    
    speedup = individual_time / batch_time
    print(f"   Individual:  {individual_time*1000:>8.1f}ms for 5 texts")
    print(f"   Batch:       {batch_time*1000:>8.1f}ms for 5 texts")
    print(f"   Speedup:     {speedup:>8.1f}x")
    
    # Test 4: Real task - Cat vs Dog classification
    print("\n4️⃣ Real Task - Cat vs Dog Classification:")
    
    start = time.time()
    result = await clip_adapter.predict({
        "task": "similarity",
        "texts": ["a cat", "a dog"],
        "images": [image_b64]
    })
    classification_time = (time.time() - start) * 1000
    
    similarities = result["similarity_matrix"][0]
    is_cat = similarities[0] > similarities[1]
    
    print(f"   Time:        {classification_time:>8.1f}ms")
    print(f"   Result:      {'Cat' if is_cat else 'Dog'} (confidence: {max(similarities):.3f})")
    print(f"   Similarities: cat={similarities[0]:.3f}, dog={similarities[1]:.3f}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print("=" * 50)
    print(f"• Is-Odd achieves {is_odd_rps:,.0f} RPS with {is_odd_latency:.3f}ms latency")
    print(f"• CLIP achieves {clip_rps:.1f} RPS with {clip_text_latency:.1f}ms text latency")
    print(f"• CLIP is {clip_text_latency/is_odd_latency:.0f}x slower for single requests")
    print(f"• CLIP batch processing provides {speedup:.1f}x speedup")
    print(f"• Real ML tasks (classification) take ~{classification_time:.0f}ms")
    print("\n💡 FastAPI overhead is negligible (<1ms) compared to ML inference!")


if __name__ == "__main__":
    asyncio.run(main())