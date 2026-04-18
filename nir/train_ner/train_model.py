import spacy
import random
from spacy.training.example import Example
from spacy.util import minibatch
from pathlib import Path
from nir.train_ner.train_data import TRAIN_DATA_2, TRAIN_DATA_EN

MODEL_NAME = "en_core_web_md"
OUTPUT_DIR = Path("assets/models")
LABELS = ["CHARACTER", "GROUP", "LOCATION", "ENVIRONMENT_ELEMENT", "ITEM", "EVENT"]
N_EPOCHS = 70
BATCH_SIZE = 4
DROPOUT = 0.25
SEED = 42

random.seed(SEED)

def normalize_entities(entities):
    normalized = []

    for ent in entities:
        if isinstance(ent, dict):
            # формат {"start": ..., "end": ..., "label": ...}
            start = ent.get("start")
            end = ent.get("end")
            label = ent.get("label")
        elif isinstance(ent, (list, tuple)) and len(ent) == 3:
            start, end, label = ent
        else:
            continue

        # защита от строк
        try:
            start = int(start)
            end = int(end)
        except:
            continue

        if start < end:
            normalized.append((start, end, label))

    return normalized

def remove_overlapping_entities(train_data):
    cleaned_data = []

    for text, annot in train_data:
        raw_entities = annot.get("entities", [])
        entities = normalize_entities(raw_entities)

        # сортируем по длине (длинные важнее)
        entities = sorted(entities, key=lambda x: (x[1] - x[0]), reverse=True)

        result = []

        for start, end, label in entities:
            overlap = False
            for r_start, r_end, _ in result:
                if not (end <= r_start or start >= r_end):
                    overlap = True
                    break

            if not overlap:
                result.append((start, end, label))

        cleaned_data.append((text, {"entities": result}))

    return cleaned_data

def fix_spacy_entities(train_data):
    fixed_data = []
    errors = []

    for text, annot in train_data:
        new_ents = []
        for start, end, label in annot.get("entities", []):
            entity_text = text[start:end]
            new_start = text.find(entity_text)

            if new_start != -1:
                new_end = new_start + len(entity_text)
                new_ents.append((new_start, new_end, label))
            else:
                lowered = text.lower()
                entity_lower = entity_text.lower().strip()

                new_start = lowered.find(entity_lower)

                if new_start != -1:
                    new_end = new_start + len(entity_lower)
                    new_ents.append((new_start, new_end, label))
                else:
                    errors.append((text, entity_text, label))

        fixed_data.append((text, {"entities": new_ents}))

    print(f"Не удалось восстановить: {len(errors)} сущностей")
    return fixed_data, errors

TRAIN_DATA_2, errors = fix_spacy_entities(TRAIN_DATA_2)
TRAIN_DATA_2 = remove_overlapping_entities(TRAIN_DATA_2)

def train_ner():
    print(f"Loading base model: {MODEL_NAME}")
    nlp = spacy.load(MODEL_NAME)

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for label in LABELS:
        ner.add_label(label)
    
    pipe_exceptions = ["ner", "trf_wordpiecer"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    print(f"Training NER with labels: {LABELS}")
    print(f"Epochs: {N_EPOCHS}, Batch size: {BATCH_SIZE}")

    nlp.config["training"]["optimizer"]["learn_rate"] = 0.001

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize()

        for epoch in range(N_EPOCHS):
            random.shuffle(TRAIN_DATA_2)
            losses = {}

            batches = minibatch(TRAIN_DATA_2, size=BATCH_SIZE)
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                nlp.update(examples, drop=DROPOUT, losses=losses)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{N_EPOCHS} - Loss: {losses.get('ner', 0):.3f}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "ner_model"
    nlp.to_disk(model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\nQuick test:")
    test_texts = [
        "Alice walked through the dark forest.",
        "The royal guards marched to the castle.",
        "Moonlight filtered through the ancient oak tree.",
        "Maria picked up the silver key.",
        "The wizard cast a spell at midnight.",
    ]
    for text in test_texts:
        doc = nlp(text)
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"  '{text}' → {ents}")

train_ner()