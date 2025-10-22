import math

def _sentiment_to_score(label: str, score: float):
    lab = (label or "").strip().lower()
    if lab.startswith("pos"):
        return float(score)
    if lab.startswith("neg"):
        return float(1.0 - score)
    return 0.5

def _to_int(px_str: str):
    try:
        return int(str(px_str).split()[0])
    except Exception:
        return 0

def confidence_interval(final_0_1, components, confidence=0.95, n_eff=30):
  
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)
    
    n_list = [n_eff] * len(components)

    variance = 0.0
    for (v, w), n in zip(components, n_list):
        v = min(1.0, max(0.0, float(v)))
        n = max(1, int(n))
        var_v = (v * (1.0 - v)) / (n + 3)
        variance += (w ** 2) * var_v

    std_error = math.sqrt(max(0.0, variance))

    lower = max(0.0, final_0_1 - z * std_error)
    upper = min(1.0, final_0_1 + z * std_error)

    return round(lower * 100.0, 1), round(upper * 100.0, 1)


def final_result(image_result, text_result, clip_result, labels_hashtag_list):
    image_analysis =  (image_result or {}).get("image_analysis", {})
    img_dim = (image_analysis or {}).get("image_dimension", {})
    width_px = _to_int(img_dim.get("width", "0"))
    height_px = _to_int(img_dim.get("height", "0"))
    size_str = img_dim.get("size", "")
    face_detected = (image_analysis or {}).get("face_detected", 0)
    min_side = min(width_px, height_px)
    dim_quality = max(0.0, min(1.0, min_side / 720.0))

    seq_item = ((text_result or {}).get("text_analysis") or [{}])[0]
    sequence_text = seq_item.get("sequence", "") or ""
    audience = seq_item.get("audience", {}) or {}
    audience_top = max(audience, key=audience.get)
    sent = seq_item.get("sentiment", {}) or {}
    sent_score = _sentiment_to_score(sent.get("label", ""), float(sent.get("score", 0.0)))
    readability = seq_item.get("readability", {}) or {}
    readability_score = float(readability.get("score", 0.0))

    clip_item = (clip_result or {}).get("clip_analysis") or {}
    seq_clip  = clip_item.get("sequence_analysis") or {}
    clip_sim = float(seq_clip.get("similarity_normalized", seq_clip.get("similarity_normalized", 0.0)))
    hashtags = (clip_item or {}).get("hashtag_analysis", []) or []
    avg_similarity_normalized = 0.0
    if hashtags:
        similarities = [float(h.get("similarity_normalized", 0.0)) for h in hashtags]
        avg_similarity_normalized = round(sum(similarities) / len(similarities), 3)
    hashtags_labels = [h.get("label") for h in hashtags if h.get("label")] or []

    hashtags_count = len(labels_hashtag_list)
    if hashtags_count >= 3:
        hashtags_note = 0.10
    elif 1 <= hashtags_count <= 2:
        hashtags_note = 0.05
    else:
        hashtags_note = 0
    
    w_clip = 0.30
    w_sent = 0.15
    w_hash = 0.10
    w_hash_len = 0.15
    w_read = 0.10
    w_face = 0.15
    w_dim  = 0.05

    face_score = 1.0 if face_detected > 0 else 0.0
    final_0_1 = (
        w_clip * clip_sim +
        w_sent * sent_score +
        w_hash * avg_similarity_normalized +
        w_hash_len * hashtags_note +
        w_read * readability_score +
        w_face * face_score +
        w_dim  * dim_quality
    )

    final_0_1 = max(0.0, min(1.0, final_0_1))
    final_score = round(final_0_1 * 100.0, 1)

    components = [
        (clip_sim, w_clip),
        (sent_score, w_sent),
        (avg_similarity_normalized, w_hash),
        (hashtags_note, w_hash_len),
        (readability_score, w_read),
        (face_score, w_face),
        (dim_quality, w_dim)
    ]
    ci_low, ci_high = confidence_interval(final_0_1, components)

    if final_score >= 80:
        final_analyse = "Excellent"
    elif final_score >= 65:
        final_analyse = "Good"
    elif final_score >= 50:
        final_analyse = "Fair"
    else:
        final_analyse = "Needs improvement"

    tips = []
    tips.append(f"Your strongest audience is: {audience_top}.")
    if face_detected == 0:
        tips.append("Consider featuring a face in the image to increase engagement.")
    if dim_quality < 0.6:
        tips.append(f"The image is relatively small ({width_px}×{height_px}, {size_str}). A larger resolution may improve perceived quality.")
    if readability_score < 0.4:
        tips.append("The caption reads as difficult; simplifying the text may improve comprehension.")
    if clip_sim < 0.6:
        tips.append("Image–caption alignment is moderate; refine the caption to better match the visual content.")
    if hashtags_count == 0:
        tips.append("No hashtags detected. Consider adding at least 3 to maximize reach.")
    elif hashtags_count < 3:
        tips.append(f"{hashtags_count} hashtag(s) detected. Consider using 3 or more for better discoverability.")
    else:
        tips.append(f"{hashtags_count} hashtags detected. Good coverage for discoverability.")
    explanation = " ".join(tips).strip()

    score_explanation = (
        f"The final score of {final_score} reflects a weighted combination of key factors: "
        f"{int(w_clip*100)}% from image–caption alignment (CLIP similarity), "
        f"{int(w_sent*100)}% from sentiment positivity, "
        f"{int(w_hash*100)}% from hashtag relevance, "
        f"{int(w_hash_len*100)}% from hashtag quantity, "
        f"{int(w_read*100)}% from caption readability, "
        f"{int(w_face*100)}% from face presence, and "
        f"{int(w_dim*100)}% from image size and quality. "
        f"And the analises of final score:"
        f"The interpretation of the final score is as follows: "
        f"Scores below 50 indicate a need for improvement, "
        f"scores between 50 and 65 are considered fair, "
        f"scores between 65 and 80 represent good performance, "
        f"and scores above 80 are classified as excellent."
    )

    return {
        "sequence": sequence_text,
        "hashtags": hashtags_labels,
        "final_analyse": final_analyse,
        "final_score": final_score,
        "confidence_interval": (ci_low, ci_high),
        "score_explanation": score_explanation,
        "tips": explanation,
        # "complete_analysis": [
        #     image_result,
        #     text_result,
        #     clip_result
        # ]
    }