#!/usr/bin/env python3
"""
Test Prithvi model loading and check memory usage.
"""

import torch

print("=" * 70)
print("PRITHVI MODEL LOADING TEST")
print("=" * 70)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("No GPU available")
print()

# Try loading Prithvi
print("Testing Prithvi model loading options...")
print("-" * 70)

# Option 1: Check if model is available
print("\n1. Checking HuggingFace model availability...")
try:
    from huggingface_hub import hf_hub_download, model_info
    
    model_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"
    info = model_info(model_id)
    print(f"   Model: {model_id}")
    print(f"   Tags: {info.tags[:5] if info.tags else 'N/A'}...")
except Exception as e:
    print(f"   Error: {e}")

# Option 2: Try loading with transformers  
print("\n2. Testing transformers AutoModel...")
try:
    from transformers import AutoModel, AutoConfig
    
    model_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"   Config loaded successfully")
    print(f"   Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
    print(f"   Num layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
except Exception as e:
    print(f"   Error loading config: {e}")

# Option 3: Try loading with timm (if it's a timm model)
print("\n3. Testing timm...")
try:
    import timm
    
    # List Prithvi models in timm
    prithvi_models = [m for m in timm.list_models() if 'prithvi' in m.lower()]
    if prithvi_models:
        print(f"   Found timm models: {prithvi_models}")
    else:
        print("   No Prithvi models found in timm")
except ImportError:
    print("   timm not installed")
except Exception as e:
    print(f"   Error: {e}")

# Option 4: Check TerraTorch
print("\n4. Testing TerraTorch (official library)...")
try:
    import terratorch
    print(f"   TerraTorch version: {terratorch.__version__}")
    
    from terratorch.models import PrithviModelFactory
    print("   PrithviModelFactory available")
except ImportError:
    print("   TerraTorch not installed. Install with: pip install terratorch")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

print("""
Based on the tests above, here are the options:

1. If TerraTorch works -> Use TerraTorch (official, recommended)
2. If transformers works -> Use with LoRA/PEFT
3. If neither works -> Use Prithvi-100M from GitHub directly
   
Let me try loading the model now...
""")

# Attempt to load
print("\nAttempting model load...")
torch.cuda.empty_cache() if torch.cuda.is_available() else None

try:
    # Try TerraTorch first
    from terratorch.models import PrithviModelFactory
    
    model = PrithviModelFactory.build_model(
        model_name="prithvi_eo_v2_300",
        task="segmentation",
        num_classes=2,
        pretrained=True
    )
    print("SUCCESS: Loaded via TerraTorch!")
    
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory used: {mem:.2f} GB")
        
except Exception as e1:
    print(f"TerraTorch failed: {e1}")
    
    try:
        # Fallback to transformers
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained(
            "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print("SUCCESS: Loaded via transformers!")
        
        if torch.cuda.is_available():
            model = model.to(device)
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory used: {mem:.2f} GB")
            
    except Exception as e2:
        print(f"Transformers failed: {e2}")
        print("\nWill need to use alternative approach (direct model loading from checkpoint)")
