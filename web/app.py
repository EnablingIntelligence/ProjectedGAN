from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Projected GAN Image Generator", version="0.1.0")
app.mount("/resources", StaticFiles(directory="resources"), name="resources")

@app.get("/", response_class=FileResponse)
def load_frontend():
    return FileResponse(path="resources/index.html", status_code=200, media_type="text/html")


@app.get("/rest/generate", response_class=StreamingResponse)
def generate_image():
    # TODO generate image
    # TODO return image
    pass
