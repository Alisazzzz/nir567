import string

import spacy
import re
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.documents import Document

def choose_merged_name(name1: str, name2: str) -> str:
    if len(name1) != len(name2):
        return name1 if len(name1) < len(name2) else name2
    return min(name1, name2, key=lambda s: s.lower())

def validate_and_clean_span(text: str, label: str, min_length: int = 2) -> tuple | None:
    cleaned = text.strip(string.punctuation + " ():,;\"'—–-")
    
    if len(cleaned) < min_length:
        return None
    if cleaned.isdigit() or cleaned.isspace():
        return None
    if re.match(r'^(why|oh|wow|yes|no|ok|the|a|an|and|but|or|in|on|at|to|for|with|by|from|is|are|was|were)$', cleaned, re.IGNORECASE):
        return None
    
    lower = cleaned.lower()
    if label == "location" and any(w in lower for w in ["key", "book", "note", "cat", "demon", "girl", "boy", "mom", "dad"]):
        label = "item" if any(w in lower for w in ["key", "book", "note"]) else "character"
    if label == "character" and len(cleaned.split()) > 3 and not any(w.istitle() for w in cleaned.split()):
        return None
    if label == "environment_element" and len(cleaned) < 4:
        return None
    
    return cleaned, label.lower()

def extract_entities_names_spacy(
    chunks: List[Document],
    ner_model_path: str,
    language: str = "en",
    min_entity_length: int = 1,
    max_entity_length: int = 50,
    preserve_all_data: bool = True, 
) -> Dict[str, Node]:

    if not Path(ner_model_path).exists():
        raise FileNotFoundError(f"NER model not found at: {ner_model_path}")
    
    nlp = spacy.load(ner_model_path)

    if "ner" not in nlp.pipe_names:
        raise ValueError(f"Model at {ner_model_path} does not contain 'ner' pipeline component")

    all_nodes: Dict[str, Node] = {}
    seen_names: Dict[str, str] = {}

    for idx, chunk in enumerate(chunks):

        print(f"[Chunk {idx+1}/{len(chunks)}] Extracting nodes names with spaCy NER.")
        
        doc = nlp(chunk.page_content)
        for ent in doc.ents:

            raw_text = ent.text.strip()
            raw_label = ent.label_
            result = validate_and_clean_span(raw_text, raw_label)
            if result is None:
                continue
            entity_name, entity_type = result

            if len(entity_name) < min_entity_length or len(entity_name) > max_entity_length:
                continue
            entity_name = clean_entity_name(entity_name)
            if not entity_name:
                continue
            if entity_type not in POSSIBLE_TYPES:
                entity_type = TYPE_CORRECTIONS.get(entity_type, "item")
            
            node_id = create_id(entity_name)

            name_key = entity_name.lower()
            if name_key in seen_names:
                existing_node = all_nodes[seen_names[name_key]]
                if chunk.metadata.get("chunk_id") not in existing_node.chunk_id:
                    existing_node.chunk_id.append(chunk.metadata.get("chunk_id"))
                continue

            node = Node(
                id=node_id,
                name=entity_name,
                type=entity_type,
                base_description="",
                base_attributes={},
                states=[],
                chunk_id=[chunk.metadata.get("chunk_id")] if chunk.metadata.get("chunk_id") is not None else []
            )
            
            all_nodes[node_id] = node
            seen_names[name_key] = node_id
            print(f"Found: '{entity_name}' [{entity_type}] (id: {node_id})")

    print(f"Extracted {len(all_nodes)} unique nodes with spaCy NER.")

    entities = extract_entities_names_ml(
        chunks=chunks,
        custom_ner_model_path=ner_model_path,
        fallback_language="en"
    )
    return entities

def merge_similar_entities_names_ml(
    nodes: List[Node],
    embedding_model,
    similarity_threshold: float = 0.85,
    exact_match_priority: bool = True,
    language: str = "en"
) -> List[Node]:

    if not nodes:
        return []
    
    merged_nodes: List[Node] = []
    for node in nodes:
        is_merged = False
        if exact_match_priority:
            for existing in merged_nodes:
                if node.type == existing.type and node.name.lower() == existing.name.lower():
                    existing.chunk_id = list(set(existing.chunk_id + node.chunk_id))
                    is_merged = True
                    print(f"Exact match: '{node.name}' '{existing.name}'")
                    break
            if is_merged:
                continue
        for idx, existing in enumerate(merged_nodes):
            if node.type != existing.type:
                continue
            sim = cosine_sim(node.name, existing.name, embedding_model)

            print(f"  • '{node.name}' vs '{existing.name}' → sim={sim:.4f}")
            
            if sim >= similarity_threshold:
                merged_name = choose_merged_name(existing.name, node.name)
                merged_nodes[idx] = Node(
                    id=existing.id,
                    name=merged_name,
                    type=existing.type,
                    base_description=existing.base_description,
                    base_attributes=existing.base_attributes,
                    states=existing.states + node.states,
                    chunk_id=list(set(existing.chunk_id + node.chunk_id))
                )            
                print(f"Merged: '{node.name}' → '{merged_name}' (sim={sim:.4f})")
                is_merged = True
                break

        if not is_merged:
            merged_nodes.append(node)
            print(f"New unique: '{node.name}'")
    
    print(f"Merging complete: {len(nodes)} → {len(merged_nodes)} nodes")

    merged_entities = merge_similar_entities_names_ml(
        nodes=nodes,
        embedding_model=embedding_model, 
        similarity_threshold=0.82,
        merge_strategy="longest"
    )
    return merged_entities


from langchain_core.documents import Document
import spacy
from spacy.matcher import Matcher
from typing import List, Dict, Optional, Tuple
import re
import logging
from pathlib import Path

from nir.graph.graph_structures import Node

def create_id(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Zа-яА-Я\s]", "", name)
    return re.sub(r"\s+", "_", cleaned.strip()).lower()


import spacy
from spacy.matcher import Matcher
from typing import List, Dict, Optional, Tuple, Set
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

POSSIBLE_TYPES = {"character", "group", "location", "environment_element", "item", "event"}

# === 1. ЛЕКСИКОНЫ ДЛЯ КОРРЕКЦИИ ТИПОВ (переопределяют модель!) ===
TYPE_OVERRIDES = {
    "character": {
        # Имена, титулы, родственные связи
        "names": {"maria", "john", "anna", "mister", "miss", "mr", "mrs", "ms", "dr", "sir", "lord", "lady"},
        "roles": {"girl", "boy", "woman", "man", "child", "daughter", "son", "mother", "father", "mom", "dad", "brother", "sister", "friend", "stranger", "hero", "villain"}
    },
    "item": {
        "animals": {"cat", "dog", "bird", "horse", "wolf", "bear", "rabbit", "fox", "demon", "angel"},  # Животные/существа как предметы, если не одушевлённые персонажи
        "objects": {"notebook", "book", "key", "sword", "rock", "bed", "door", "window", "lamp", "torch", "potion", "coin", "gem", "ring", "map", "compass"},
        "rooms": {"bedroom", "kitchen", "bathroom", "hall", "cellar", "attic", "room"}
    },
    "environment_element": {
        "parts": {"wall", "ceiling", "floor", "path", "trail", "entrance", "exit", "stairs", "roof", "chimney", "fence", "gate", "corner", "window", "door"}
    },
    "event": {
        "nominalizations": {"battle", "arrival", "departure", "death", "meeting", "fight", "journey", "wedding", "funeral", "ceremony", "attack", "rescue", "discovery"}
    }
}

# === 2. СТОП-ЛИСТЫ ДЛЯ ОТБРАКОВКИ ===
REJECT_PATTERNS = {
    "contains_pronoun": {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their"},
    "generic_verbs": {"be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "tell", "let", "lets", "go", "going", "come", "look", "looking", "sit", "sitting", "stay", "staying", "walk", "walking", "run", "running", "appear", "appeared", "found", "lost", "found"},
    "abstract_nouns": {"world", "fact", "history", "details", "event", "gameplay", "while", "stuff", "thing", "part", "place", "time", "way", "day", "night", "life", "system", "process", "state", "scene", "finish", "finishes", "cutie", "dear", "ancient", "greatest", "little", "angel"}
}

VALID_DEPS = {"nsubj", "dobj", "pobj", "attr", "compound", "ROOT", "appos", "conj", "nmod", "poss"}

def should_reject_span(text: str, first_token: Optional[spacy.tokens.Token] = None) -> Tuple[bool, Optional[str]]:
    """Проверяет, стоит ли отбросить кандидата. Возвращает (reject, reason)."""
    text_lower = text.lower().strip()
    
    # 1. Слишком короткое / длинное
    if len(text_lower) < 2: return True, "too_short"
    
    # 2. Содержит местоимение (кроме случаев, когда это всё слово)
    tokens = set(re.findall(r'\b\w+\b', text_lower))
    if tokens & REJECT_PATTERNS["contains_pronoun"] and len(tokens) > 1:
        return True, "contains_pronoun"
    
    # 3. Первый токен — глагол из стоп-листа
    if first_token and first_token.lemma_.lower() in REJECT_PATTERNS["generic_verbs"]:
        return True, "generic_verb"
    
    # 4. Всё словосочетание — абстрактное существительное
    if text_lower in REJECT_PATTERNS["abstract_nouns"]:
        return True, "abstract_noun"
    
    # 5. Содержит предлог/союз (значит, это фраза, а не сущность)
    if re.search(r'\b(of|and|or|but|in|on|at|to|for|with|by|from|into|upon|without)\b', text_lower):
        return True, "contains_preposition"
    
    return False, None

def correct_entity_type(text: str, predicted_type: str, first_token: Optional[spacy.tokens.Token] = None) -> Tuple[str, str]:
    """Корректирует тип сущности на основе лексиконов. Возвращает (new_type, reason)."""
    text_lower = text.lower().strip()
    lemma = first_token.lemma_.lower() if first_token else text_lower
    
    # Приоритет: явные совпадения в словарях
    for target_type, categories in TYPE_OVERRIDES.items():
        for category, words in categories.items():
            if text_lower in words or lemma in words:
                return target_type, f"lexicon_override:{category}"
    
    # Эвристики по суффиксам/префиксам
    if predicted_type != "event" and lemma in TYPE_OVERRIDES["event"]["nominalizations"]:
        return "event", "event_nominalization"
    
    if predicted_type != "environment_element" and text_lower in TYPE_OVERRIDES["environment_element"]["parts"]:
        return "environment_element", "env_part"
    
    # Если модель предсказала character, но слово — животное/предмет → item
    if predicted_type == "character" and (lemma in TYPE_OVERRIDES["item"]["animals"] or text_lower in TYPE_OVERRIDES["item"]["objects"]):
        return "item", "animal_object_override"
    
    return predicted_type, "no_override"

def extract_entities_names_ml(
    chunks: List[Document],
    custom_ner_model_path: Optional[str] = None,
    fallback_language: str = "en",
    label_mapping: Optional[Dict[str, str]] = None,
    min_entity_length: int = 2,
    max_entity_length: int = 50,
    confidence_threshold: float = 0.60,
    use_custom_ner: bool = True,  # Можно отключить модель полностью для тестов
    preserve_all_data: bool = True,
) -> Dict[str, Node]:

    # === Загрузка модели ===
    if use_custom_ner and custom_ner_model_path and Path(custom_ner_model_path).exists():
        logger.info(f"Loading custom NER model: {custom_ner_model_path}")
        nlp = spacy.load(custom_ner_model_path)
    else:
        model_name = f"{fallback_language}_core_web_md"
        logger.info(f"Using fallback model: {model_name}")
        nlp = spacy.load(model_name)

    # === Добавляем недостающие пайпы ===
    missing_pipes = [p for p in ["tagger", "parser"] if p not in nlp.pipe_names]
    if missing_pipes:
        try:
            base_nlp = spacy.load(f"{fallback_language}_core_web_sm")
            for pipe in missing_pipes:
                if pipe in base_nlp.pipe_names:
                    nlp.add_pipe(pipe, source=base_nlp)
        except:
            logger.warning("Could not add parser/tagger. Dependency filtering disabled.")

    # === Маппинг лейблов ===
    DEFAULT_MAPPING = {
        "PERSON": "character", "CHARACTER": "character", "CHAR": "character", "HERO": "character",
        "ORG": "group", "GROUP": "group", "FACTION": "group",
        "GPE": "location", "LOC": "location", "FAC": "location", "LOCATION": "location",
        "PRODUCT": "item", "ITEM": "item", "OBJECT": "item", "ANIMAL": "item",
        "EVENT": "event", "ACTION": "event",
        "ENV": "environment_element", "ENV_ELEMENT": "environment_element"
    }
    final_mapping = {**DEFAULT_MAPPING}
    if label_mapping:
        final_mapping.update(label_mapping)
    final_mapping = {k.upper(): v for k, v in final_mapping.items()}

    # === Matcher для дополнения ===
    matcher = Matcher(nlp.vocab)
    fallback_patterns = {
        "environment_element": [
            [{"LOWER": {"IN": list(TYPE_OVERRIDES["environment_element"]["parts"])}}]
        ],
        "item": [
            [{"LOWER": {"IN": list(TYPE_OVERRIDES["item"]["objects"])}}],
            [{"POS": "NOUN", "LEMMA": {"IN": list(TYPE_OVERRIDES["item"]["rooms"])}}]
        ]
    }
    for label, patterns in fallback_patterns.items():
        for i, pat in enumerate(patterns):
            matcher.add(f"{label.upper()}_{i}", [pat])

    all_nodes: Dict[str, Node] = {}
    seen_names: Dict[str, str] = {}

    for idx, chunk in enumerate(chunks):
        logger.info(f"[Chunk {idx+1}/{len(chunks)}] Extracting...")
        doc = nlp(chunk.page_content)
        chunk_id = chunk.metadata.get("chunk_id")
        candidates = {}

        # === A. Кастомный NER (с низким доверием!) ===
        if use_custom_ner and "ner" in nlp.pipe_names:
            for ent in doc.ents:
                raw_label = ent.label_.upper()
                predicted_type = final_mapping.get(raw_label, None)
                if not predicted_type or predicted_type not in POSSIBLE_TYPES:
                    continue
                
                # === ФИЛЬТРАЦИЯ СПАНОВ ===
                reject, reason = should_reject_span(ent.text, ent[0])
                if reject:
                    logger.debug(f"Rejected by NER filter: '{ent.text}' ({reason})")
                    continue
                
                # === КОРРЕКЦИЯ ТИПА ===
                final_type, override_reason = correct_entity_type(ent.text, predicted_type, ent[0])
                
                key = (ent.text.lower(), final_type)
                # Базовая уверенность для кастомной модели — НИЗКАЯ!
                base_conf = 0.60
                if ent[0].pos_ == "PROPN": base_conf += 0.1
                if override_reason != "no_override": base_conf += 0.1  # Лексикон подтверждает
                
                candidates[key] = {
                    "text": ent.text,
                    "type": final_type,
                    "confidence": base_conf,
                    "source": f"custom_ner:{raw_label}",
                    "token": ent[0],
                    "override": override_reason
                }

        # === B. Правила (как независимый источник) ===
        matches = matcher(doc)
        for match_id, start, end in matches:
            label = nlp.vocab.strings[match_id].split("_")[0].lower()
            span = doc[start:end]
            
            reject, reason = should_reject_span(span.text, span[0])
            if reject:
                continue
            
            final_type, override_reason = correct_entity_type(span.text, label, span[0])
            key = (span.text.lower(), final_type)
            
            rule_conf = 0.65 if span[0].pos_ == "PROPN" else 0.60
            
            if key in candidates:
                # NER уже предложил → усредняем, но не повышаем слепо
                existing = candidates[key]
                if existing["source"].startswith("custom_ner"):
                    # Если тип совпадает — чуть повышаем уверенность
                    if existing["type"] == final_type:
                        candidates[key]["confidence"] = min(existing["confidence"] + 0.05, 0.85)
                        candidates[key]["source"] += "+rule"
            elif rule_conf >= confidence_threshold:
                candidates[key] = {
                    "text": span.text,
                    "type": final_type,
                    "confidence": rule_conf,
                    "source": "rule",
                    "token": span[0],
                    "override": override_reason
                }

        # === C. Создание Node ===
        for key, cand in candidates.items():
            if cand["confidence"] < confidence_threshold: continue
            if len(cand["text"]) < min_entity_length or len(cand["text"]) > max_entity_length: continue

            entity_name = re.sub(r'^[^\w]+|[^\w]+$', '', cand["text"]).strip()
            if not entity_name: continue

            node_id = create_id(entity_name)
            name_key = entity_name.lower()

            if name_key in seen_names:
                existing_node = all_nodes[seen_names[name_key]]
                if chunk_id and chunk_id not in existing_node.chunk_id:
                    existing_node.chunk_id.append(chunk_id)
                if cand["confidence"] > existing_node.base_attributes.get("confidence", 0):
                    existing_node.base_attributes.update({
                        "confidence": round(cand["confidence"], 3),
                        "source": cand["source"],
                        "override": cand.get("override")
                    })
                continue

            node = Node(
                id=node_id,
                name=entity_name,
                type=cand["type"],
                base_description="",
                base_attributes={
                    "confidence": round(cand["confidence"], 3),
                    "source": cand["source"],
                    "pos": cand["token"].pos_ if cand["token"] else "",
                    "dep": cand["token"].dep_ if cand["token"] else "",
                    "override": cand.get("override", "none")
                },
                states=[],
                chunk_id=[chunk_id] if chunk_id else []
            )
            
            all_nodes[node_id] = node
            seen_names[name_key] = node_id

    logger.info(f"Extracted {len(all_nodes)} unique entities.")
    return all_nodes

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import Levenshtein  # pip install python-Levenshtein


def compute_hybrid_similarity(
    name1: str, 
    name2: str, 
    embedding_model,
    type1: str,
    type2: str,
    weight_embedding: float = 0.6,
    weight_lexical: float = 0.3,
    weight_morpho: float = 0.1
) -> float:
    """Комбинированная метрика сходства: эмбеддинги + лексика + морфология."""
    
    # 1. Эмбеддинг-сходство (косинус)
    try:
        emb1 = embedding_model.encode([name1])[0]
        emb2 = embedding_model.encode([name2])[0]
        emb_sim = cosine_similarity([emb1], [emb2])[0][0]
    except:
        emb_sim = 0.0
    
    # 2. Лексическое сходство (Levenshtein + Jaccard)
    name1_norm = name1.lower().strip()
    name2_norm = name2.lower().strip()
    
    lev_sim = 1 - (Levenshtein.distance(name1_norm, name2_norm) / max(len(name1_norm), len(name2_norm), 1))
    
    tokens1 = set(re.findall(r'\w+', name1_norm))
    tokens2 = set(re.findall(r'\w+', name2_norm))
    jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2) if (tokens1 | tokens2) else 0
    lexical_sim = (lev_sim + jaccard) / 2
    
    # 3. Морфологическое сходство (леммы + тип)
    morpho_sim = 1.0 if type1 == type2 else 0.0
    if type1 == type2 and name1_norm != name2_norm:
        # Дополнительный бонус, если одно является подстрокой другого
        if name1_norm in name2_norm or name2_norm in name1_norm:
            morpho_sim = 0.9
    
    # Взвешенная сумма
    total_sim = (
        weight_embedding * emb_sim +
        weight_lexical * lexical_sim +
        weight_morpho * morpho_sim
    )
    
    return total_sim


def choose_merged_name(name1: str, name2: str, priority_rules: Optional[Dict] = None) -> str:
    """Выбирает каноническое имя при слиянии."""
    # Приоритет: длиннее → с заглавной → без артиклей
    candidates = [name1, name2]
    
    # Если есть правила приоритета по типу
    if priority_rules and name1.lower() in priority_rules:
        return name1
    if priority_rules and name2.lower() in priority_rules:
        return name2
    
    # Эвристики
    candidates.sort(key=lambda x: (-len(x), -x[0].isupper(), x.lower()))
    return candidates[0]


def merge_similar_entities_names_ml(
    nodes: List[Node],
    embedding_model,
    similarity_threshold: float = 0.82,  # Чуть ниже, т.к. используем гибридную метрику
    exact_match_priority: bool = True,
    language: str = "en",
    merge_strategy: str = "longest"  # "longest", "first", "canonical"
) -> List[Node]:
    
    if not nodes:
        return []
    
    # Сортируем по уверенности (если есть в атрибутах) и длине имени
    nodes_sorted = sorted(
        nodes, 
        key=lambda n: (-n.base_attributes.get("confidence", 0.5), -len(n.name), n.name.lower())
    )
    
    merged_nodes: List[Node] = []
    merge_log = []
    
    for node in nodes_sorted:
        is_merged = False
        
        # 1. Точное совпадение (быстрый путь)
        if exact_match_priority:
            for existing in merged_nodes:
                if (node.type == existing.type and 
                    node.name.lower().strip() == existing.name.lower().strip()):
                    # Объединяем chunk_id и состояния
                    existing.chunk_id = list(set(existing.chunk_id + node.chunk_id))
                    existing.states = list(set(existing.states + node.states))
                    merge_log.append(f"Exact: '{node.name}' → '{existing.name}'")
                    is_merged = True
                    break
            if is_merged:
                continue
        
        # 2. Гибридное сходство
        best_match_idx = None
        best_sim = 0.0
        
        for idx, existing in enumerate(merged_nodes):
            if node.type != existing.type:
                continue  # Не сливаем разные типы
            
            sim = compute_hybrid_similarity(
                node.name, existing.name, 
                embedding_model,
                node.type, existing.type
            )
            
            if sim > best_sim:
                best_sim = sim
                best_match_idx = idx
        
        if best_sim >= similarity_threshold and best_match_idx is not None:
            existing = merged_nodes[best_match_idx]
            
            # Выбираем каноническое имя
            if merge_strategy == "longest":
                merged_name = max(existing.name, node.name, key=len)
            elif merge_strategy == "first":
                merged_name = existing.name
            else:  # canonical
                merged_name = choose_merged_name(existing.name, node.name)
            
            # Объединяем данные
            merged_node = Node(
                id=existing.id,  # Сохраняем ID первого (канонического)
                name=merged_name,
                type=existing.type,
                base_description=existing.base_description or node.base_description,
                base_attributes={
                    **existing.base_attributes,
                    **node.base_attributes,
                    "merged_from": [existing.name, node.name],
                    "merge_similarity": round(best_sim, 4)
                },
                states=list(set(existing.states + node.states)),
                chunk_id=list(set(existing.chunk_id + node.chunk_id))
            )
            
            merged_nodes[best_match_idx] = merged_node
            merge_log.append(f"Merged: '{node.name}' → '{merged_name}' (sim={best_sim:.4f})")
            is_merged = True
        
        if not is_merged:
            merged_nodes.append(node)
            merge_log.append(f"New: '{node.name}' [{node.type}]")
    
    # Логирование
    for log_entry in merge_log:
        logger.debug(log_entry)
    
    logger.info(f"Merging complete: {len(nodes)} → {len(merged_nodes)} nodes")
    return merged_nodes


def extract_graph_with_ner(
        chunks: List[Document],       
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        graph_class = NetworkXGraph,
        preserve_all_data: bool = True,
        language: str = "en"
    ) -> KnowledgeGraph:
    
    NER_MODEL_PATH = "assets/models/ner_model"
    graph = graph_class()

    nodes_names = extract_entities_names_spacy(
        chunks=chunks,
        ner_model_path=NER_MODEL_PATH,
        language=language,
        preserve_all_data=preserve_all_data
    )
    print("---NODES---")
    for node in nodes_names.values():
        print(node)

    merged_nodes_names = merge_similar_entities_names_ml(
        nodes=list(nodes_names.values()),
        embedding_model=embedding_model,
        similarity_threshold=0.85,
        language=language
    )
    print("---MERGED NODES---")
    print(merged_nodes_names)

    nodes, edges = extract_graph_info(
        chunks=chunks,
        nodes=merged_nodes_names,
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )

    completed_nodes, completed_edges = complete_graph(
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )

    all_nodes = [n for n in completed_nodes.values()]
    all_edges = [e for e in completed_edges.values()]

    merged_result = merge_similar_nodes(
        chunks=chunks,
        nodes=all_nodes, 
        edges=all_edges, 
        llm=llm, 
        embedding_model=embedding_model,
        preserve_all_data=preserve_all_data,
        language=language
    )
    for n in merged_result[0].values():
        graph.add_node(n)
    for e in merged_result[1].values():
        if e.source and e.target:
            graph.add_edge(e)

    print(f"Graph built with {len(all_nodes)} nodes and {len(all_edges)} edges.") #DEBUGGING
    
    nodes_in_graph = graph.get_all_nodes()
    edges_in_graph = graph.get_all_edges()
    events_impacts = extract_events_impact(
        chunks=chunks, 
        nodes=nodes_in_graph, 
        edges=edges_in_graph, 
        llm=llm,
        preserve_all_data=preserve_all_data,
        language=language
    )  

    events_only = [node for node in nodes_in_graph if node.type == "event"]
    for event in events_impacts:
        for event_in_graph in events_only:
            print (event_in_graph.name, event.event_name)
            if event_in_graph.name == event.event_name:
                apply_event_impact_on_graph(graph, event, event_in_graph)

    return graph