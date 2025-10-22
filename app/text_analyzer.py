import os
from typing import Any, Dict, List
from transformers import pipeline
import textstat

from app.logger import LogManager

class TextAnalyzer:
    def __init__(self, post_text_list: list[str]):

        log_manager = LogManager('textAnalyzer')
        self.logger = log_manager.get_logger()
        
        self.post_text_list = post_text_list

        zero_shot_model = os.getenv("ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
        sentiment_model = os.getenv("SENTIMENT_MODEL", "finiteautomata/bertweet-base-sentiment-analysis")
        keyphrase_model = os.getenv("KEYPHRASE_MODEL", "ml6team/keyphrase-extraction-kbir-inspec")
        
        self.model_classifier = pipeline(
            "zero-shot-classification", 
            model=zero_shot_model
        )

        self.model_sentiment_analysis = pipeline(
            "sentiment-analysis", 
            model = sentiment_model)
        
        self.model_key_word = pipeline(
            "token-classification",
            model=keyphrase_model,
            aggregation_strategy="simple"
        )
        

    def classifier_public_age(self):

        labels = [
            "young female audience (18–30)",
            "young male audience (18–30)",
            "adult audience (30–50)",
            "general audience"
        ]
        
        return self.model_classifier(self.post_text_list, candidate_labels=labels)
    
    def sentiment_analysis(self):
        
        result = self.model_sentiment_analysis(self.post_text_list)
        label_map = {"POS": "positivo", "NEG": "negativo", "NEU": "neutro"}
        mapped_results: List[Dict[str, Any]] = []

        for text, res in zip(self.post_text_list, result):
            label = label_map.get(res.get("label", ""), res.get("label", ""))
            score = round(float(res.get("score", 0.0)), 3)

            mapped_results.append({
                "sequence": text,
                "label": label,
                "score": score
            })

        return mapped_results
    
    def key_word_analyse(self):

        raw_results = self.model_key_word(self.post_text_list)
        mapped_results: List[Dict[str, Any]] = []

        for text, kw_list in zip(self.post_text_list, raw_results):
            key_words = []
            for kw in kw_list:
                word = (kw.get("word") or kw.get("label") or "").strip()
                if not word:
                    continue
                score = round(float(kw.get("score", 0.0)), 3)
                key_words.append({
                    "label": word,
                    "score": score
                })

            key_words.sort(key=lambda x: x["score"], reverse=True)

            mapped_results.append({
                "sequence": text,
                "key_words": key_words
            })

        return mapped_results
    
    def readability_metrics(self):
        mapped_results: List[Dict[str, Any]] = []

        for text in self.post_text_list:
            fre = textstat.flesch_reading_ease(text)               

            fre_clamped = max(0.0, min(100.0, float(fre)))
            ease01 = round(fre_clamped / 100.0, 3)

            if ease01 >= 0.8:
                level = "very easy"
            elif ease01 >= 0.6:
                level = "easy"
            elif ease01 >= 0.4:
                level = "moderate"
            elif ease01 >= 0.2:
                level = "hard"
            else:
                level = "very hard"
        
            mapped_results.append({
                "sequence": text,
                "readability": ease01,
                "level": level
            })

        return mapped_results
    
    def analyser(self) -> Dict[str, Any]:
        self.logger.info("Starting TextAnalyzer.analyser orchestrator.")
        try:
            audience_result = self.classifier_public_age()
        except Exception as e:
           self.logger(f"audience classification failed: {e}", exc_info=True)

        try:
            sentiment_result = self.sentiment_analysis()
        except Exception as e:
            self.logger(f"sentiment analysis failed: {e}", exc_info=True)

        try:
            key_word_result = self.key_word_analyse()
        except Exception as e:
            self.logger(f"keyphrase extraction failed: {e}", exc_info=True)

        try:
            readability_metrics_result = self.readability_metrics()
        except Exception as e:
            self.logger(f"readability metrics failed: {e}", exc_info=True)

        try:
            self.logger.info("Merging analysis outputs.")
            merged: Dict[str, Dict[str, Any]] = {}

            def get_entry(seq: str) -> Dict[str, Any]:
                if seq not in merged:
                    merged[seq] = {"sequence": seq}
                return merged[seq]

            if isinstance(audience_result, list):
                for item in audience_result:
                    seq = item.get("sequence")
                    if not seq:
                        continue
                    entry = get_entry(seq)
                    audience_dict = {
                        label: round(float(score), 3)
                        for label, score in zip(item.get("labels", []), item.get("scores", []))
                    }
                    entry["audience"] = audience_dict
            else:
                merged["_audience_error"] = audience_result

            if isinstance(sentiment_result, list):
                for item in sentiment_result:
                    seq = item.get("sequence")
                    if not seq:
                        continue
                    entry = get_entry(seq)
                    entry["sentiment"] = {
                        "label": item.get("label"),
                        "score": round(float(item.get("score", 0.0)), 3),
                    }
            else:
                merged["_sentiment_error"] = sentiment_result

            if isinstance(key_word_result, list):
                for item in key_word_result:
                    seq = item.get("sequence")
                    if not seq:
                        continue
                    entry = get_entry(seq)
                    entry["key_words"] = item.get("key_words", [])
            else:
                merged["_keyphrase_error"] = key_word_result

            if isinstance(readability_metrics_result, list):
                for item in readability_metrics_result:
                    seq = item.get("sequence")
                    if not seq:
                        continue
                    entry = get_entry(seq)
                    entry["readability"] = {
                        "score": round(float(item.get("readability", 0.0)), 3),
                        "level": item.get("level"),
                    }
            else:
                merged["_readability_error"] = readability_metrics_result

            text_analysis = [v for k, v in merged.items() if not k.startswith("_")]

            self.logger.info(
                f"Merging completed. Sequences: {len(text_analysis)}. "
            )
          
            return {"text_analysis": text_analysis}

        except Exception as e:
            self.logger.error(f"Failed to merge analysis outputs: {e}", exc_info=True)
            return {"error": f"merge failed: {e}"}