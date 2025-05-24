from transformers import pipeline
import wikipedia

def merge_consecutive_entities(ner_results):
    merged = []
    seen_texts = set()
    current = None

    for token in sorted(ner_results, key=lambda x: x["index"]):
        word = token["word"]
        entity = token["entity"]
        index = token["index"]

        # Remove subword prefix if present
        if word.startswith("##"):
            word = word[2:]

        if current is None:
            current = {
                "entity": entity,
                "text": word,
                "start": token["start"],
                "end": token["end"],
                "prev_index": index
            }
        elif (
            entity == current["entity"]
            and (index == current["prev_index"] + 1)
        ):
            if token["word"].startswith("##"):
                current["text"] += word
            else:
                current["text"] += " " + word
            current["end"] = token["end"]
            current["prev_index"] = index
        else:
            if current["text"] not in seen_texts:
                merged.append({
                    "entity": current["entity"],
                    "text": current["text"],
                    "start": current["start"],
                    "end": current["end"]
                })
                seen_texts.add(current["text"])
            current = {
                "entity": entity,
                "text": word,
                "start": token["start"],
                "end": token["end"],
                "prev_index": index
            }

    # Add the last one
    if current and current["text"] not in seen_texts:
        merged.append({
            "entity": current["entity"],
            "text": current["text"],
            "start": current["start"],
            "end": current["end"]
        })

    return merged


def search_keywords_wiki(keyword):
    wiki_text = wikipedia.summary(wikipedia.search(keyword)[0], sentences=1, auto_suggest=False)
    return wiki_text

def replace_keywords(model, text):
    ner = model(text)
    keywords = merge_consecutive_entities(ner)

    for keyword in keywords:
        try:
            wiki_text = search_keywords_wiki(keyword['text'])
            text = text.replace(keyword['text'], wiki_text)
        except Exception as e:
            continue

    return text


if __name__ == "__main__":
    # Load the NER pipeline
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    
    caption = """The image highlights the controversy surrounding Morocco's withdrawal from hosting the 2015 African Cup of Nations (AFCON) due to Ebola fears and its impact on CAF's decision-making process. It depicts CAF President Issa Hayatou, a central figure in this decision, alongside unidentified men in a formal setting, likely a conference room discussing the situation. Hayatou's attire suggests importance, while the image's slightly distorted quality hints at the tension and uncertainty surrounding the event. The location within the room, the white wall and black door, emphasizes the confined and decisive nature of the discussion. This snapshot captures a pivotal moment in AFCON history, showcasing the challenges faced by African nations in organizing major sporting events amidst global health concerns and the complex web of political and logistical considerations involved in such decisions."""
    
    enhanced_caption = replace_keywords(ner_pipeline, caption)
    print(f"Enhanced caption: {enhanced_caption}")
