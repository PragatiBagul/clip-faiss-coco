"""
app/main.py

Minimal FastAPI app that mounts the generation router and initializes
a LlamaGenerator instance at startup for reuse across requests.

Assumes:
- router at app.routers.generation exists and exports `router`
- llama.inference.inference.LlamaGenerator exists
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import generation as generation_router_module

# import model wrapper
try:
    from llama.inference.inference import LlamaGenerator
except Exception as e:
    LlamaGenerator = None
    logging.warning(f"Could not import LlamaGenerator at startup: {e}")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="clip-faiss-coco (with LLaMA generation)", version="0.2")

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount routers
app.include_router(generation_router_module.router)

# expose generator globally via app.state
@app.on_event("startup")
async def startup_event():
    # create a singleton LlamaGenerator instance and attach to app.state
    base_model = os.environ.get("LLAMA_BASE_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    use_8bit = os.environ.get("LLAMA_USE_8BIT", "false").lower() in ("1", "true", "yes")
    lora_path = os.environ.get("LLAMA_LORA_ADAPTER", "") or None

    if LlamaGenerator is None:
        logger.warning("LlamaGenerator class not available. Generation endpoints may fail.")
        app.state.llama_generator = None
        return

    try:
        logger.info(f"Initializing LlamaGenerator (model={base_model}, use_8bit={use_8bit}, lora={lora_path})")
        gen = LlamaGenerator(base_model=base_model, use_8bit=use_8bit, lora_adapter_path=lora_path)
        app.state.llama_generator = gen
        # also set the generator in the router module if it expects global access
        try:
            generation_router_module.GENERATOR = gen
        except Exception:
            pass
    except Exception as e:
        logger.exception(f"Failed to initialize LlamaGenerator on startup: {e}")
        app.state.llama_generator = None

@app.on_event("shutdown")
async def shutdown_event():
    gen = getattr(app.state, "llama_generator", None)
    if gen:
        try:
            gen.close()
        except Exception:
            pass
