from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import httpx
import os
import base64
import vtracer
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Vector AI Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SVG_SYSTEM_PROMPT = """You are an expert SVG vector graphics designer.
Your job is to generate clean, minimal, scalable SVG code based on the user's prompt.

STRICT RULES:
- Output ONLY raw SVG code. No explanation, no markdown, no backticks.
- Always start with <svg and end with </svg>
- Always use viewBox="0 0 100 100"
- Use only basic SVG shapes: circle, rect, path, line, polyline, polygon, ellipse, text
- No external fonts, no images, no scripts inside SVG
- Keep paths simple and clean
- Use stroke and fill attributes directly on elements
- No background rectangle unless specifically asked
- Make it look professional and minimal"""


class PromptRequest(BaseModel):
    prompt: str
    style: str = "icon"  # icon | logo | illustration


class SVGResponse(BaseModel):
    svg: str
    method: str


@app.get("/")
def root():
    return {"status": "Vector AI Generator is running"}


@app.post("/generate-svg", response_model=SVGResponse)
async def generate_svg(req: PromptRequest):
    """Generate SVG directly using Groq LLM"""
    try:
        style_hints = {
            "icon": "Simple, single-color or two-color icon. Clean lines. Suitable for UI use.",
            "logo": "Logo mark. Bold, geometric, memorable. Works at small and large sizes.",
            "illustration": "Detailed vector illustration. Multiple colors allowed. Artistic style."
        }
        hint = style_hints.get(req.style, style_hints["icon"])

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SVG_SYSTEM_PROMPT},
                {"role": "user", "content": f"Style: {hint}\n\nGenerate SVG for: {req.prompt}"}
            ],
            max_tokens=2000,
            temperature=0.3
        )

        svg_code = response.choices[0].message.content.strip()

        # Clean up markdown fences if model added them
        for fence in ["```svg", "```xml", "```"]:
            svg_code = svg_code.replace(fence, "")
        svg_code = svg_code.strip()

        # Basic validation
        if not svg_code.startswith("<svg"):
            raise HTTPException(status_code=500, detail="Model did not return valid SVG")

        return SVGResponse(svg=svg_code, method="groq-llm")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq error: {str(e)}")


@app.post("/generate-image-svg", response_model=SVGResponse)
async def generate_image_svg(req: PromptRequest):
    """Generate image via Pollinations.ai then convert to SVG using vtracer"""
    try:
        # Step 1: Build Pollinations URL (no API key needed)
        safe_prompt = req.prompt.replace(" ", "%20")
        pollinations_url = (
            f"https://image.pollinations.ai/prompt/{safe_prompt}"
            f"?width=512&height=512&nologo=true&enhance=true"
        )

        # Step 2: Fetch the image
        async with httpx.AsyncClient(timeout=60.0) as http:
            img_response = await http.get(pollinations_url)

        if img_response.status_code != 200:
            raise HTTPException(status_code=502, detail="Pollinations API failed")

        # Step 3: Convert to PNG with PIL (ensure correct format)
        image = Image.open(io.BytesIO(img_response.content)).convert("RGB")

        # Resize to manageable size for vtracer
        image = image.resize((256, 256), Image.LANCZOS)

        png_buffer = io.BytesIO()
        image.save(png_buffer, format="PNG")
        png_bytes = png_buffer.getvalue()

        # Step 4: Convert PNG to SVG using vtracer
        svg_code = vtracer.convert_raw_image_to_svg(
            png_bytes,
            img_format="png",
            colormode="color",       # color | binary
            hierarchical="stacked",  # stacked | cutout
            mode="spline",           # spline | polygon | none
            filter_speckle=4,        # higher = cleaner but less detail
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3
        )

        return SVGResponse(svg=svg_code, method="pollinations-vtracer")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image-to-SVG error: {str(e)}")


@app.get("/health")
def health():
    groq_key_set = bool(os.getenv("GROQ_API_KEY"))
    return {
        "status": "ok",
        "groq_key_configured": groq_key_set
    }


POST /generate-svg
  → Takes: { "prompt": "sun icon" }
  → Groq LLM generates SVG code
  → Returns: { "svg": "<svg>...</svg>" }

POST /generate-image
  → Takes: { "prompt": "sun icon" }
  → Calls Pollinations.ai free API
  → Converts PNG to SVG via vtracer
  → Returns: { "svg": "..." }