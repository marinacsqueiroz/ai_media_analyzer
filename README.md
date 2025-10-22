# 🧠 AI Post Analyzer

A complete **multimodal post analysis system** combining **text and image** using artificial intelligence.  
It leverages 🤗 [Hugging Face Transformers](https://huggingface.co/transformers) and computer vision to evaluate **sentiment**, **readability**, **target audience**, **image quality**, and **text–image alignment (CLIP)**.

---

## 📁 Project Structure

```
app/
├── logs/                    
│   └── __init__.py
├── clip_analyser.py         # CLIP-based image–text similarity analysis
├── image_analyzer.py        # Image analysis (dimensions, faces)
├── text_analyzer.py         # Text analysis (sentiment, readability, keywords)
├── final_results.py         # Combines all results and calculates final score
├── logger.py                # Centralized logging system
├── main.py                  # FastAPI app entry point
├── requirements.txt         # Project dependencies
├── show.json                # Example of input/output
└── test_api.py              # Tests and API integration
```

---

## ⚙️ Features

### TextAnalyzer
Performs **linguistic and semantic analysis** using NLP models.

- Target audience classification (age and gender)
- Sentiment analysis (positive, neutral, negative)
- Keyword extraction based on relevance
- Readability metrics (Flesch Reading Ease)

**Models used:**
- facebook/bart-large-mnli → zero-shot classification  
- finiteautomata/bertweet-base-sentiment-analysis → sentiment  
- ml6team/keyphrase-extraction-kbir-inspec → keyword extraction  

---

### ImageAnalyzer
Analyzes an image from a URL and provides:

- Dimensions and size (in pixels and KB)
- Face detection using OpenCV (`haarcascade_frontalface_default.xml`)

**Libraries:**  
cv2, numpy, urllib

---

### ClipAnalyzer
Applies the **CLIP model (Contrastive Language–Image Pretraining)** to measure semantic similarity between text and image.

It computes embeddings and similarity levels (High, Medium, Low) between the post caption and the visual content.

**Default model:** `openai/clip-vit-base-patch32`

---

### final_results.py
Consolidates outputs from all analyzers and computes a **final engagement score (0–100)**, based on weighted metrics:

| Factor | Weight |
|--------|---------|
| Image–caption alignment (CLIP) | 30% |
| Sentiment positivity | 15% |
| Hashtag relevance | 10% |
| Hashtag quantity | 15% |
| Caption readability | 10% |
| Face presence | 15% |
| Image size/quality | 5% |

Also generates:
- Confidence interval
- Final classification (Excellent, Good, Fair, Needs improvement)
- Actionable tips for optimization

---

### logger.py
Manages logs per module and automatically creates timestamped `.log` files inside `/app/logs/<subfolder>/YYYY-MM-DD.log`.

**Example log output:**
```
[INFO] Starting TextAnalyzer.analyser orchestrator.
[INFO] Running hashtag analysis...
[ERROR] Error during face detection analysis: ...
```

---

### main.py — FastAPI Server
Provides an API endpoint for full post analysis.

**Endpoint:**  
`POST /analyze-post`

**Request body:**
```json
{
  "text": ["Your caption with #hashtags"],
  "image_url": "https://example.com/image.jpg"
}
```

**Response example:**
```json
{
  "sequence": "Your caption with hashtags",
  "hashtags": ["#AI", "#Innovation"],
  "final_analyse": "Good",
  "final_score": 73.4,
  "confidence_interval": [68.1, 78.6],
  "tips": "Consider featuring a face in the image to increase engagement."
}
```

---

## 🚀 How to Run Locally

1. Clone this repository and open the folder.  
2. Create and activate a virtual environment:  
   ```bash
   python -m venv .venv && .venv\Scripts\activate
   ```  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
4. Run the API:  
   ```bash
   python -m uvicorn app.main:app --reload
   ```  
5. Open http://localhost:8000/docs to test.

---


