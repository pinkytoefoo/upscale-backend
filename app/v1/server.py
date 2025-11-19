from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
# from workers import WorkerEntrypoint

# class Default(WorkerEntrypoint):
#     async def fetch(self, request):
#         import asgi

#         return await asgi.fetch(app, request, self.env)

app = FastAPI()
# TODO: add middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Or ["*"] to allow all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
#     allow_headers=["*"],  # Allow all headers
# )

def _api_ai_upscaler(image, scale_factor) -> np.ndarray:
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    model_path = f"models/ESPCN_x{scale_factor}.pb"
    sr.readModel(model_path)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("espcn", scale_factor)

    # Upscale the image
    result = sr.upsample(image)
    # print(np.asarray(Image.open(image_path)))
    # print(cv2.imread(image_path))
    return result

@app.get("/")
async def api():
    return {"api": "working😁"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upscale/")
async def upscale(file: UploadFile = File(...), scale_factor: int = Query(2, description="Scaling factor. Could be '2' or '4'")):
    if not file:
        raise HTTPException(status_code=400, detail="File not uploaded")

    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    final_image: np.ndarray = _api_ai_upscaler(image, scale_factor)

    _, img_encoded = cv2.imencode(".png", final_image) 

    return Response(content=img_encoded.tobytes(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)