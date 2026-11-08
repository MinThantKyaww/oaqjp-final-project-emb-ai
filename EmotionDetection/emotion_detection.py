"""Detect emotions with the IBM Skills Network Watson NLP service."""

import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_detector(text_to_analyze):
    """Return the five emotion scores and the dominant emotion."""
    response = requests.post(
        URL,
        json={"raw_document": {"text": text_to_analyze}},
        headers=HEADERS,
        timeout=30,
    )
    if response.status_code == 400:
        return dict.fromkeys((*EMOTIONS, "dominant_emotion"))

    response.raise_for_status()
    prediction = response.json()["emotionPredictions"][0]["emotion"]
    scores = {emotion: prediction[emotion] for emotion in EMOTIONS}
    scores["dominant_emotion"] = max(EMOTIONS, key=prediction.__getitem__)
    return scores
