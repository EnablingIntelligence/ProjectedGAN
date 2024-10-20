from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from io import BytesIO


from gan import ImageGenerator

image_generator = ImageGenerator(ckpt_path="./gan/ckpt/generator.pth")

app = FastAPI(title="Projected GAN Image Generator", version="0.1.0")
app.mount("/resources", StaticFiles(directory="resources"), name="resources")


@app.get("/", response_class=FileResponse)
def load_frontend():
    return FileResponse(
        path="resources/index.html", status_code=200, media_type="text/html"
    )


@app.get("/rest/generate", response_class=Response)
def generate_image():
    outstream = BytesIO()
    image = image_generator.generate()
    image.save(outstream, format="PNG")

    return Response(content=outstream.getvalue(), media_type="image/png")
