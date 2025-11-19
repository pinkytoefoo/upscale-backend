# TODO: abstract

from fastapi import File, UploadFile, Response, HTTPException, Query
import cv2
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
from PIL import Image
from fastapi import APIRouter

router = APIRouter()

thread_pool = ThreadPoolExecutor(max_workers=4)

# Cache model instances to avoid reloading
MODEL_CACHE = {}

class CachedSuperResModel:
    def __init__(self):
        self.sr = None
        self.scale_factor = None
        self.model_type = "espcn"
    
    def load_model(self, scale_factor: int):
        if self.sr is None or self.scale_factor != scale_factor:
            model_path = f"models/ESPCN_x{scale_factor}.pb"
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(model_path)
            self.sr.setModel("espcn", scale_factor)
            self.scale_factor = scale_factor
        return self.sr

def get_cached_model() -> CachedSuperResModel:
    """Get or create cached model instance"""
    if "super_res" not in MODEL_CACHE:
        MODEL_CACHE["super_res"] = CachedSuperResModel()
    return MODEL_CACHE["super_res"]

async def api_ai_upscaler(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """Upscale image using thread pool to avoid blocking"""
    loop = asyncio.get_event_loop()
    
    def upscale_sync():
        model_wrapper = get_cached_model()
        sr = model_wrapper.load_model(scale_factor)
        return sr.upsample(image)
    
    # Run in thread pool to avoid blocking event loop
    result = await loop.run_in_executor(thread_pool, upscale_sync)
    return result

def optimize_memory_usage(image: np.ndarray) -> np.ndarray:
    """Reduce memory usage by optimizing image format"""
    # Convert to more memory-efficient format if possible
    if image.dtype == np.float64:
        image = image.astype(np.float32)
    return image

def validate_image_size(image: np.ndarray, max_pixels: int = 4000000) -> bool:
    """Prevent processing overly large images"""
    height, width = image.shape[:2]
    return height * width <= max_pixels

@router.get("/")
async def api():
    return {"api": "working😁"}

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/upscale/")
async def upscale(
    file: UploadFile = File(...), 
    scale_factor: int = Query(2, ge=2, le=4, description="Scaling factor. Could be '2' or '4'")
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read file in chunks to avoid memory issues with large files
    file_bytes = await file.read()
    
    # Use PIL for more efficient image decoding
    try:
        image = Image.open(io.BytesIO(file_bytes))
        # Convert to numpy array maintaining RGB order (OpenCV uses BGR)
        image_np = np.array(image)
        
        # If image has alpha channel, remove it for processing
        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
            
        # Convert RGB to BGR for OpenCV
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        print(f"Image decoding failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Validate image size
    if not validate_image_size(image_np):
        raise HTTPException(
            status_code=400, 
            detail="Image too large. Maximum allowed dimensions: 4000x1000 pixels"
        )

    # Optimize memory usage
    image_np = optimize_memory_usage(image_np)

    try:
        final_image = await api_ai_upscaler(image_np, scale_factor)
        
        # Encode to JPEG for smaller file size, use PNG for lossless
        is_photo = len(np.unique(final_image)) > 256  # Simple check if it's a photo
        encode_format = ".jpg" if is_photo else ".png"
        media_type = "image/jpeg" if is_photo else "image/png"
        
        _, img_encoded = cv2.imencode(encode_format, final_image, 
                                    [cv2.IMWRITE_JPEG_QUALITY, 95] if is_photo else [])
        
        del image_np, final_image
        
        return Response(content=img_encoded.tobytes(), media_type=media_type)
        
    except Exception as e:
        print(f"Upscaling failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Upscaling failed")
    