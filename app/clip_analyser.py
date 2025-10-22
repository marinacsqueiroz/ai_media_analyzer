import os
import re
import torch
from transformers import pipeline
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict

from app.logger import LogManager


class ClipAnalyzer:
    def __init__(self, post_text_list, image_path, labels_hashtag_list):

        log_manager = LogManager('ClipAnalyzer')
        self.logger = log_manager.get_logger()

        clip_model = os.getenv("KEYPHRASE_MODEL", "openai/clip-vit-base-patch32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        labels_hashtag = " ".join(labels_hashtag_list)
        self.labels_hashtag = re.findall(r"#\w+", labels_hashtag)

        self.post_text_list = post_text_list
        self.image_path = image_path
        self.model_clip = pipeline(
            task="zero-shot-image-classification",
            model=clip_model,
            dtype=torch.bfloat16,
            device=0
            )
        self.model_clip_pre_trained = CLIPModel.from_pretrained(clip_model)
        self.model_clip_pre_trained = self.model_clip_pre_trained.to(device)
        self.model_clip_processor = CLIPProcessor.from_pretrained(clip_model)
    
    def labels_analyse(self):
        return self.model_clip(self.image_path, candidate_labels=self.post_text_list)
    
    def hashtag_analyse(self):        
        return self.model_clip(self.image_path, candidate_labels=self.labels_hashtag)
    
    def embeddings_text_image(self, text_list):
        inputs = self.model_clip_processor(text=text_list, images=[self.image_path], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            img_emb = self.model_clip_pre_trained.get_image_features(**inputs)
            txt_emb = self.model_clip_pre_trained.get_text_features(**inputs)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        cosine = (txt_emb @ img_emb.T).squeeze()
        sim01  = (cosine + 1) / 2

        cosine_list = cosine.tolist()
        sim_list = sim01.tolist()
        
        if type(cosine_list) == float:
            cosine_list = [cosine_list]
        if type(sim_list) == float:
            sim_list = [sim_list]
        
        results = []
        for i, (cos, sim) in enumerate(zip(cosine_list, sim_list)):
            sequence = text_list[i]

            if sim > 0.85:
                level = "High"
            elif sim > 0.6:
                level = "Medium"
            else:
                level = "Low"

            result = {
                "label": sequence,
                "cosine_similarity": round(cos, 3),
                "similarity_normalized": round(sim, 3),
                "evaluation": level
            }
            results.append(result)

        return results
        
    def analyser(self):
        self.logger.info("🚀 Starting CLIP analysis pipeline...")

        try:
            self.logger.info("Step : Running hashtag analysis...")
            hashtags_scores = self.hashtag_analyse()
            self.logger.info(f"Hashtag analysis completed. Found {len(hashtags_scores) if hashtags_scores else 0} items.")

            self.logger.info("Step 2: Running label (sequence) analysis...")
            sequences_scores = self.labels_analyse()
            self.logger.info(f"Sequence analysis completed. Found {len(sequences_scores) if sequences_scores else 0} items.")

            self.logger.info("Step 3: Computing CLIP embeddings for hashtags...")
            clip_hashtag_metrics = self.embeddings_text_image(self.labels_hashtag)
            self.logger.info(f"CLIP embeddings for hashtags computed. Count: {len(clip_hashtag_metrics)}")

            self.logger.info("Step 4: Computing CLIP embeddings for text sequences...")
            clip_sequence_metrics = self.embeddings_text_image(self.post_text_list)
            self.logger.info(f"CLIP embeddings for sequences computed. Count: {len(clip_sequence_metrics)}")
        except Exception as e:
            self.logger.error(f"Error during initial analysis steps: {e}", exc_info=True)
            raise

        # Build lookup tables
        clip_by_label = {
            d["label"]: {k: d[k] for k in ("cosine_similarity", "similarity_normalized", "evaluation")}
            for d in clip_hashtag_metrics
        }
        clip_seq_by_text = {
            d["label"]: {k: d[k] for k in ("cosine_similarity", "similarity_normalized", "evaluation")}
            for d in clip_sequence_metrics
        }

        hashtags_out = []
        self.logger.info("Merging hashtag similarity scores with CLIP metrics...")
        for item in (hashtags_scores or []):
            lab = item.get("label")
            if not lab:
                continue

            merged = {
                "label": lab,
                "image_text_similarity_score": round(float(item.get("score", 0.0)), 3),
            }

            if lab in clip_by_label:
                merged.update(clip_by_label[lab])
                self.logger.debug(f"Merged CLIP metrics for hashtag: {lab}")
            else:
                self.logger.warning(f"No CLIP metrics found for hashtag: {lab}")

            hashtags_out.append(merged)

        hashtags_out.sort(key=lambda x: x.get("image_text_similarity_score", 0.0), reverse=True)
        self.logger.info(f"Hashtag results merged and sorted ({len(hashtags_out)} total).")

        # Best sequence analysis
        self.logger.info("Determining best sequence based on image-text similarity...")
        if sequences_scores:
            best_seq = max(sequences_scores, key=lambda x: x.get("score", 0.0))
            seq_text = best_seq.get("label", "")
            sequence_out = {
                "text": seq_text,
                "image_text_similarity_score": round(float(best_seq.get("score", 0.0)), 3),
            }

            if seq_text in clip_seq_by_text:
                sequence_out.update(clip_seq_by_text[seq_text])
                self.logger.info(f"Best sequence matched with CLIP metrics: {seq_text}")
            else:
                sequence_out.update({
                    "cosine_similarity": 0.0,
                    "similarity_normalized": 0.0,
                    "evaluation": "No CLIP metrics available for this sequence."
                })
                self.logger.warning(f"No CLIP metrics found for best sequence: {seq_text}")
        else:
            self.logger.warning("No sequence scores found. Using default empty output.")
            sequence_out = {
                "text": "",
                "image_text_similarity_score": 0.0,
                "cosine_similarity": 0.0,
                "similarity_normalized": 0.0,
                "evaluation": "No sequence data available."
            }

        result = {
            "clip_analysis": {
                "hashtag_analysis": hashtags_out,
                "sequence_analysis": sequence_out
            }
        }

        self.logger.info("CLIP analysis completed successfully.")
        self.logger.debug(f"Final result keys: {list(result['clip_analysis'].keys())}")

        return result
