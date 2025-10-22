

import re
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from app.clip_analyser import ClipAnalyzer
from app.final_results import final_result
from app.image_analyzer import ImageAnalyzer
from app.text_analyzer import TextAnalyzer
from app.logger import LogManager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou coloque o domínio específico do seu front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: List[str] = Field(default_factory=list)
    image_url: str


@app.post("/analyze-post")
def read_root(req: AnalyzeRequest) -> Dict[str, Any]:

    text_raw = req.text
    image_path = req.image_url

    text_raw = " ".join(req.text) if isinstance(req.text, list) else req.text

    labels_hashtag_list = re.findall(r"#\w+", text_raw)

    clean_text = re.sub(r"#\w+", "", text_raw).strip()
    post_text_list = [clean_text] if clean_text else []

    log_manager = LogManager('mainLog')
    logger = log_manager.get_logger()

    logger.info("Initializing analyzers (ClipAnalyzer, TextAnalyzer, ImageAnalyzer).")

    try:
        clip_analyzer = ClipAnalyzer(
            post_text_list=post_text_list,
            image_path=image_path,
            labels_hashtag_list=labels_hashtag_list
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to init ClipAnalyzer: {e}")
    
    try:
        text_analyzer = TextAnalyzer(post_text_list=post_text_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to init TextAnalyzer: {e}")

    try:
        image_analyzer = ImageAnalyzer(image_path=image_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to init ImageAnalyzer: {e}")

    try:
        logger.info("Starting ClipAnalyzer.analyser()")
        clip_result = clip_analyzer.analyser()
        logger.info("ClipAnalyzer.analyser() finished successfully.")
    except Exception as e:
        logger.error(f"ClipAnalyzer.analyser() failed: {e}")
        clip_result = {"error": f"Clip analysis failed: {e}"}

    try:
        logger.info("Starting TextAnalyzer.analyser()")
        text_result = text_analyzer.analyser()
        logger.info("TextAnalyzer.analyser() finished successfully.")
    except Exception as e:
        logger.error(f"TextAnalyzer.analyser() failed: {e}")
        text_result = {"error": f"Text analysis failed: {e}"}

    try:
        logger.info("Starting ImageAnalyzer.analyser()")
        image_result = image_analyzer.analyser()
        logger.info("ImageAnalyzer.analyser() finished successfully.")
    except Exception as e:
        logger.error(f"ImageAnalyzer.analyser() failed: {e}")
        image_result = {"error": f"Image analysis failed: {e}"}

    return final_result(image_result, text_result, clip_result, labels_hashtag_list)

