#All prompts for graph extraction are here


SYSTEM_PROMPT_ENTITIES_EN = """
    Extract entities and relations from the text fragment.
    INPUT:
        - Text fragment
        - Coreference clusters (groups of mentions for the same entity)

    RULES:
        1. Extract all meaningful entities. Resolve pronouns using coreference. If there exist entitities that is not mentioned in coreference clusters, but exist in text, extract them too.
        2. EVENTS:  extract only if they impact game world or are named.
        3. PARENTHETICAL NAMES: "Princess (Elly)" → name="Elly", type="character".
        4. NAME: short canonical form, no descriptions (e.g., "Elly", not "Elly the Princess").
    
    STRUCTURES AND RULES.
      1. Entities (nodes). For each entity identified from a coreference cluster or found by yourself, output an object with the following fields:
        - "name": entity name (designation). A name is a personal name, without any additional information and descriptions (even if it is included into coreference clusters: chose the one without additional info, the shortest one).
        - "type": one of the allowed types described below. Be sure that living creatures are characters!
        - "base_description": additional information and descriptions about the entity. This field answer the question "What this entity is and how it can be described?" as fully as possible, but based ONLY on an available information. You can copy here all words or sentences that describe this entity in the input text.
        - "base_attributes": dictionary of attributes; attributes are some characteristics of the entity that can describe it. For example, if there is an entity chair, and this chair is wooden, there will be attribute "material" : "wood". 
        IMPORTANT: for entities of type "event" attribute "time" is indispensable: it is a string describing time of an event, answers the question "when did this event take place?" ("in the evening", "1042 b.c", "in the Age of the Dragon", etc.). ONLY if time cannot be extracted, this string may be empty: "".
      2. Relations (edges). For every relation found:
        - "node1": name of the first entity. Be careful: DO NOT produce None in this fields, add an entity if it is needed here. Answers the question "Who or what has a connection with another entity?".
        - "node2": name of the second entity. Be careful: DO NOT produce None in this fields, add an entity if it is needed here. Answers the question "Who or what has a connection with node1 entity?".
        - "relation_from1to2": lowercase verb or short phrase, describing relation between node1 and node2. MUST NOT be null or None or Empty. Answers the question "How the FIRST entity connected to the SECOND entity?"
        - "relation_from2to1": lowercase verb or short phrase, describing inverted relation between nodes: from node2 to node1. MUST NOT be null or None or Empty. Answers the question "How the SECOND entity connected to the FIRST entity?". For example, if A "holds" B, then B "is held by" A.
        - "description": additional information, detailed description for relation, describing this connection as fully as possible.
        - "weight": float (default 1.0). Answers the question "How strong are these two entities connected by this relation?". For example, two characters can be friends with weight 1.0 - best friends, and friends with weight 0.3 - almost do not friends, only familiar to each other people.
        IMPORTANT: Relationships must always have ONE node1 and ONE node2. If multiple node1/node2 are implied, create multiple edges.
    
    ENTITY TYPES. Use exactly the following types for the "type" field:
      1. "character" — a sentient being or individual acting within the narrative. Can have different relations.
      2. "group" — a collection of characters acting as a unit. This entities can have "located in", "take part in" (an event), "contains" (a character, and the character "is a part of") and other different edges with other types and between nodes of this type.
      3. "location" — a geographical or spatial setting. Between entities of this type there should be edges describing spatial relations like "connected with", "located to the north/south/east/west of", "has a road to", etc.
      4. "environment_element" — a part or feature of a location. MUST have a relation "located in", which connects it to a certain location where this element is located.
      5. "item" — a physical object that can be possessed or interacted with. Can have different relations.
      6. "event" — an action, occurrence, or change of state. Events form the underlying chronological and causal structure (fabula), and they must have "time" field in "base_attributes".
      IMPORTANT FOR EVENTS: try to extract as much information as possible about chronological order of events: between entities of this type MUST be chronological relations like "precedes" and "follows".

    OUTPUT:
    <reasoning> 
        Briefly explain what you extracted and why. Under 80 words.
    </reasoning>
    {{
        "nodes": [
            {{
                "name": "short canonical name",
                "type": "character | group | location | environment_element | item | event",
                "base_description": "what this entity is, based only on text",
                "base_attributes": {{"key": "value"}}
            }}
        ],
        "edges": [
            {{
                "node1": "entity name",
                "node2": "entity name",
                "relation_from1to2": "lowercase verb phrase",
                "relation_from2to1": "lowercase inverse verb phrase",
                "description": "context of connection: what does it mean",
                "weight": "float"
            }}
        ]
    }}

    ### EXAMPLE (DO NOT USE IT IN OUTPUT) ###
    Input Text: "In summer 1670, Alice entered the dark forest."
    Coreference: [["Alice", "she"], ["dark forest", "the forest"]]

    <reasoning>
    Found character Alice, location Dark Forest, and event of entering. Time is "summer 1670".
    </reasoning>
    {{
        "nodes": [
            {{"name": "Alice", "type": "character", "base_description": "A girl", "base_attributes": {{}}}},
            {{"name": "Dark Forest", "type": "location", "base_description": "A forest", "base_attributes": {{}}}},
            {{"name": "Alice enters forest", "type": "event", "base_description": "Alice enters the forest", "base_attributes": {{"time": "summer 1670"}}}}
        ],
        "edges": [
            {{"node1": "Alice enters forest", "node2": "Alice", "relation_from1to2": "involves", "relation_from2to1": "participates in", "description": "Alice participates in event", "weight": 1.0}},
            {{"node1": "Alice enters forest", "node2": "Dark Forest", "relation_from1to2": "occurs in", "relation_from2to1": "contains event", "description": "Event occurs in forest", "weight": 1.0}}
        ]
    }}
    ### END EXAMPLE ###

    CRITICAL:
        - Output ONLY <reasoning> block followed by JSON. No other text.
        - For events: "time" in base_attributes is mandatory.
        - Relations must have both directions, non-empty strings.
    
"""

SYSTEM_PROMPT_MERGING_EN = """
    Determine if two nodes represent the SAME real-world entity.
    INPUT: Two node objects with name, base_description, base_attributes.

    DECISION RULES:
        1. MERGE if: same name OR same description OR clearly same entity in context. Ask: "Are these names/descriptions synonyms or referring to one entity?"
        2. DO NOT MERGE if: similar type but different instances (e.g., "a sword" vs "the king's sword" without confirmation).
        3. When merging: combine descriptions and attributes. Prefer more specific values.
        4. NAME-IN-DESCRIPTION: If A.name appears in B.base_description → very likely same entity → MERGE.

    OUTPUT:
    <reasoning>
        Briefly explain why merge or not.
    </reasoning>
    {{
        "name": "best chosen name",
        "base_description": "combined description",
        "base_attributes": {{"key": "value"}}
    }}
    If not merging → all fields empty: {{"name": "", "base_description": "", "base_attributes": {{}} }}

    CRITICAL: Output ONLY <reasoning> block followed by JSON. No other text.
"""

SYSTEM_PROMPT_MERGING_IN_GRAPH_EN = """
    Merge two nodes into one node, preserve as much data as possible. Create for two nodes combined base_description, merge list of attributes and list of states: for states, be careful:
        1. if two states from different nodes are similar - create combined state.
        2. if two states from different nodes are different (have different times), you must in this case only copy states into final merged node. 
        3. also be careful with time_start_event and time_end_event: copy them carefully, and while merging, pay attention to when the states change (so time_end_event for the previous state is time_start_event for the next state)

    INPUT: Two node objects with name, base_description, base_attributes and states list.

    OUTPUT FORMAT:
    <reasoning> 
        Explain how you will combine two nodes. Under 80 words.
    </reasoning>
    {{
          "name": "string (choose best name)",
          "type": "string (one of types copied from input nodes)",
          "base_description": "string (combined description)",
          "base_attributes": {{"key": "value"}}
          "states": {{
                {{ 
                    "sid": "string (id for state copied from input or created if states were merged)", 
                    "current_description": "string (combined description from two states or copied description)",
                    "current_attributes": {{"key": "value"}},
                    "time_start_event": "string (event id copied from input)",
                    "time_end_event": "string (event id copied from input)"
                }}
          }}
    }}
    CRITICAL: Output ONLY <reasoning> block followed by JSON. No other text.
"""

SYSTEM_PROMPT_EVENTS_IMPACTS_EN = """
    Identify how EVENTS change ENTITIES and RELATIONS.
    INPUT:
        - Text fragment that mentions evenst (one or more)
        - Events names mentioned in text fragments
        - List of entities (with IDs) mentioned
        - List of relations (with IDs) mentioned

    RULES:
        1. Focus on STATE CHANGES only. For every mentioned event, ask: "Before event: entity was X. After event: is it still X?"
        2. For affected nodes: provide full new description (be sure, that this description describes an entity as fully as possible for this state, including facts, that were known before) and copy base attributes with changes for the period AFTER event. 
        3. If there is NO STATES in the entity, and only in that case, create BEFORE state: you must SHOW CHANGES in this entity. In this case, use "before EVENT_NAME" structure for event name in output array. If you create a prior state and a resulting state, the prior state MUST have time_end_event equal to the current event.
        4. If in node's states you see that there is a state without time_end_event and one of events you are working currently with is next state after opened one (without time_end_event), add this opened state into changed_states and write time_end_event for it. Only last state for node can be opened. Do not add "before" state, if there exist some states already for this entity: prefer changing states with time_end_event.
        Be very carefull: if you have more than one event and creating more than one state for an entity, you must realise, which event is earlier and will have before-state, and which event will only close previous state and add new after-state. 
        5. For affected edges: mark time_start_event (relation begins) or time_end_event (relation ends).
        6. Use EXACT IDs from input lists. Do not invent new IDs.
        7. If nothing changes → return empty lists.
    If you've added BEFORE event, be sure that you add event itself too.

    OUTPUT:
    <reasoning>
        For every event, explain its impact on nodes and entities very shorly, under 100 words totally. Answer the question, if you should add before-event state. Keep this block as short as possible.
    </reasoning>
    {{
        "events_with_impact": [
            {{
                "event_name": "event name from list" or "before event name from list",
                "changed_states": [
                    {{
                        "node_id" : "MUST match entity ID from input",
                        "sid" : "State ID COPIED WITHOUT CHANGES from input state that you want to modify",
                        "time_end_event": "string"
                    }}
                ]
                "affected_nodes": [
                    {{
                        "id": "MUST match entity ID from input",
                        "name": "string",
                        "new_current_description": "full description after event or before event",
                        "new_current_attributes": {{"key": "value"}},
                        "time_start_event": "string or null",
                        "time_end_event": "string or null"
                    }}
                ],
                "affected_edges": [
                    {{
                        "id": "MUST match edge ID from input",
                        "new_description": "string",
                        "time_start_event": "string or null",
                        "time_end_event": "string or null"
                    }}
                ]
            }}
        ]
    }}

    ### EXAMPLE (DO NOT USE IT IN OUTPUT) ###
    Input:
    Text: "The king died. His son became the new ruler."
    Events: ["The king died"]
    Entities: [
        {{"id": "char_king", "name": "King", "type": "character", "states": 
            {{
                "sid": "king_becomes_father_king",
                "current_description": "King of the kingdom. Now has a son.",
                "current_attributes": {{ "status": "alive", "family_state" : "father" }},
                "time_start_event": "king_becomes_father",
                "time_end_event": null,
            }}
        }}, 
        {{"id": "char_son", "name": "Prince", "type": "character"}}
    ]
    Relations: [
        {{"id": "edge_01", "source": "char_son", "target": "char_king", "relation": "is son of"}}
    ]

    Output:
    <reasoning>
        King's state changed from alive to dead. Son's state changed from prince to ruler. No relations changed. Prince entity does not have states, so I must add "before EVENT_NAME" structure.
    </reasoning>
    {{
        "events_with_impact": [
            {{
                "event_name": "before The king died",
                "changed_states" : [],
                "affected_nodes": [
                    {{
                        "id": "char_king",
                        "name": "King",
                        "new_current_description": "The king if the kingdom",
                        "new_current_attributes": {{ "status": "alive" }},
                        "time_start_event": null,
                        "time_end_event": "The king died"
                    }},
                    {{
                        "id": "char_son",
                        "name": "Prince",
                        "new_current_description": "The son of the ruler of the kingdom",
                        "new_current_attributes": {{ "title": "prince" }},
                        "time_start_event": null,
                        "time_end_event": "The king died"
                    }}
                ],
                "affected_edges": []
            {{
                "event_name": "The king died",
                "changed_states" : [
                    {{
                        "node_id": "char_king",
                        "sid": "king_becomes_father_king",
                        "time_end_event": "The king died"
                    }}
                ],
                "affected_nodes": [
                    {{
                        "id": "char_king",
                        "name": "King",
                        "new_current_description": "The deceased king",
                        "new_current_attributes": {{ "status": "dead" }},
                        "time_start_event": "The king died",
                        "time_end_event": null
                    }},
                    {{
                        "id": "char_son",
                        "name": "Prince",
                        "new_current_description": "The new ruler of the kingdom",
                        "new_current_attributes": {{ "title": "king" }},
                        "time_start_event": "The king died",
                        "time_end_event": null
                    }}
                ],
                "affected_edges": []
            }}
        ]
    }}
    ### END EXAMPLE ###

    CRITICAL: Output ONLY <reasoning> block followed by JSON. No other text. Use exact IDs from input.
"""

SYSTEM_PROMPT_GRAPH_COMPLETION_EN = """
    Complete a narrative knowledge graph by finding missed entities and relations. Try to find more than 2-4 extra relations.
    PRIMARY FOCUS: MAXIMIZE CONNECTIVITY THROUGH RELATIONS. Your main goal is to recover missed edges. Extract entities ONLY if they are strictly necessary to form valid, text-supported relations.

    INPUT:
        - Text fragment
        - Existing entities
        - Existing relations

    RULES:
    1. CHECK FIRST: Compare with existing lists. Do NOT duplicate.
    2. RELATION-FIRST EXTRACTION:
        - Scan text for connections: spatial, possession, participation, causal, temporal, social, state changes.
        - If a meaningful connection is missing, extract it and reversed connection. Create the entity only if it does not exist yet.
        - NEVER leave an extracted entity isolated. Every new entity must connect to at least one existing node.
    3. EVENT CONNECTIVITY:
        - Events MUST link to: participants, locations, causes, effects, and TEMPORAL ORDER (precedes/follows).
    4. FORMATING:
        For missing_entities:
            - name: Short canonical name (e.g., "cat" not "the black cat")
            - type: One of: character, group, location, environment_element, item, event
            - base_description: What this entity is (1-2 sentences)
            - base_attributes: Dictionary of characteristics (e.g., {{"color": "black", "material": "wood"}})
            - reason: Why this entity was missed and evidence from text
            - chunk_reference: Exact quote from text mentioning this entity

        For missing_relations:
            - node1: Name of first entity (must match existing or new entity name)
            - node2: Name of second entity (must match existing or new entity name)
            - relation_from1to2: Verb phrase, lowercase (e.g., "is on", "owns", "participates in")
            - relation_from2to1: Inverse verb phrase, lowercase (e.g., "has on", "is owned by", "involves")
            - description: Context of this connection
            - weight: Float 0.0-1.0 (1.0 = strong/explicit, 0.5 = implied)
            - reason: Why this relation was missed and evidence from text
            - chunk_reference: Exact quote from text implying this relation

    5. EVIDENCE (MANDATORY):
        - reason: Explain why it was missed and how the text supports it.
        - chunk_reference: DIRECT QUOTE from the text. NO PARAPHRASING. If exact quote is long, truncate with "...".

    OUTPUT FORMAT:
    <reasoning>
        Briefly summarize missed entities, recovered relations, and key connectivity improvements. Less then 80 words.
    </reasoning>
    {{
        "missing_entities": [
            {{
                    "name": "short canonical name",
                    "type": "character | group | location | environment_element | item | event",
                    "base_description": "1-2 sentences describing this entity, may be from text",
                    "base_attributes": {{"key": "value"}},
                    "reason": "string",
                    "chunk_reference": "exact quote"
            }}
        ],
        "missing_relations": [
            {{
                "node1": "entity name",
                "node2": "entity name",
                "relation_from1to2": "lowercase verb",
                "relation_from2to1": "lowercase inverse verb",
                "description": "context of connection",
                "weight": "float",
                "reason": "string",
                "chunk_reference": "exact quote"
            }}
        ]
    }}

    ### EXAMPLE (DO NOT USE IT IN OUTPUT) ###
    Input:
    Text: "The black cat slept on the wooden bed in the bedroom. Mary watched her pet from the doorway."
    Entities: [
        {{"name": "bed", "type": "item", "description": "A wooden bed"}}, 
        {{"name": "bedroom", "type": "location", "description": "A room"}}
    ]
    Existing Relations: []

    Output:
    <reasoning>
    Found 2 missing entities: cat (character) and Mary (character). Found 3 missing relations: cat-on-bed, bed-in-bedroom, Mary-watches-cat. The cat was mentioned but not extracted. Mary was mentioned by name but not extracted. Spatial relations were not captured.
    </reasoning>
    {{
        "missing_entities": [
            {{
                "name": "cat",
                "type": "character",
                "base_description": "A black cat who is Mary's pet",
                "base_attributes": {{"color": "black", "owner": "Mary"}},
                "reason": "Cat is a sentient being (character) mentioned in text but not extracted",
                "chunk_reference": "The black cat slept on the wooden bed"
            }},
            {{
                "name": "Mary",
                "type": "character",
                "base_description": "A person who owns the cat and watches it",
                "base_attributes": {{"role": "pet owner"}},
                "reason": "Mary is a character mentioned by name but not extracted",
                "chunk_reference": "Mary watched her pet from the doorway"
            }}
        ],
        "missing_relations": [
            {{
                "node1": "cat",
                "node2": "bed",
                "relation_from1to2": "is on",
                "relation_from2to1": "has on",
                "description": "The cat is sleeping on the bed",
                "weight": 1.0,
                "reason": "Explicit spatial relation 'on' was not extracted",
                "chunk_reference": "cat slept on the wooden bed"
            }},
            {{
                "node1": "bed",
                "node2": "bedroom",
                "relation_from1to2": "is in",
                "relation_from2to1": "contains",
                "description": "The bed is located in the bedroom",
                "weight": 1.0,
                "reason": "Explicit spatial relation 'in' was not extracted",
                "chunk_reference": "bed in the bedroom"
            }},
            {{
                "node1": "Mary",
                "node2": "cat",
                "relation_from1to2": "owns",
                "relation_from2to1": "is owned by",
                "description": "Mary owns the cat as a pet",
                "weight": 1.0,
                "reason": "Possession relation implied by 'her pet' was not extracted",
                "chunk_reference": "Mary watched her pet"
            }}
        ]
    }}
    ### END EXAMPLE ###
    
    CRITICAL:
        - Output ONLY <reasoning> block followed by JSON. No other text.
        - Every new item MUST contain chunk_reference.
        - Prioritize relations over entities. Isolated entities are invalid.
"""

SYSTEM_PROMPT_ENTITIES_NAMES_EN = """
    You have to accuratly extract all entities from a text fragment, and do not miss any of entities.
    TASK: Extract entity names and types from text fragment.

    INPUT:
        - Text fragment
        - Coreference clusters

    RULES:
        1. Extract all meaningful entities. Resolve pronouns using coreference. If there exist entitities that is not mentioned in coreference clusters, but exist in text, extract them too - DO NOT rely ONLY on coreference clusters.
        2. EVENTS:  extract only if they impact game world or are named.
        3. PARENTHETICAL NAMES: "Princess (Elly)" → name="Elly", type="character".
        4. NAME: short canonical form, no descriptions (e.g., "Elly", not "Elly the Princess").

    ENTITY TYPES (use exactly these):
        1. "character" — a sentient being or individual acting within the narrative.
        2. "group" — a collection of characters acting as a unit.
        3. "location" — a geographical or spatial setting.
        4. "environment_element" — a part or feature of a location. Must have "located in" attribute.
        5. "item" — a physical object that can be possessed or interacted with.
        6. "event" — an action, occurrence, or change of state. Events form the underlying chronological and causal structure (fabula). 

    OUTPUT:
    <reasoning> 
        Briefly explain extraction, for every event shortly (three words maximum) describe if it needed to be extracted.
    </reasoing>
    {{
        "nodes": [
            {{
                "name": "short canonical name",
                "type": "character | group | location | environment_element | item | event"
            }}
        ]
    }}

    ### EXAMPLE (DO NOT USE IT IN OUTPUT) ###
    Input Text: "In summer 1670, Alice entered the dark forest."
    Coreference: [["Alice", "she"], ["dark forest", "the forest"]]

    <reasoning>
    Found character Alice, location Dark Forest, and event of entering.
    </reasoning>
    {{
        "nodes": [
            {{"name": "Alice", "type": "character" }},
            {{"name": "Dark Forest", "type": "location" }},
            {{"name": "Alice enters forest", "type": "event" }}
        ]
    }}
    ### END EXAMPLE ###

    CRITICAL: Output ONLY <reasoning> block followed by JSON. No other text.
"""

SYSTEM_PROMPT_MERGING_NAMES_EN = """
    Determine if two entity names refer to the SAME entity.
    INPUT: Two names and their contexts.

    RULES:
        1. MERGE if: names are synonyms or clearly refer to same entity in context. Ask: "Do these names describe one entity?"
        2. DO NOT MERGE if: contexts indicate different entities.
        3. When merging: create single best name.

    OUTPUT:
        1. First write <reasoning> block: explain decision, why the nodes are similar or why they are different.
        2. Then output JSON:
        {{"name": "best chosen name"}}
        If not merging: {{"name": ""}}

    CRITICAL: Output ONLY <reasoning> block followed by JSON. No other text.
"""

SYSTEM_PROMPT_ENTITIES_WITH_NAMES_EN = """
    Enrich pre-extracted entities with information from text and extract relations between them (as many relations as possible).
    INPUT:
        - Text fragment
        - List of entities mentioned in this text fragment with names, types and base description that was already extracted.
    RULES:
    1. ENTITIES: Fill out all required fields for all entities mentioned in input. If there is base_description and base_attributes, combine it with information extracted from current text. Try do not lost any of information.
    2. RELATIONS: Extract BOTH directions for every connection:
        - relation_from1to2: How node1 connects to node2 (e.g., "holds")
        - relation_from2to1: Inverse relation (e.g., "is held by")
    3. RELATIONS: Relationships must always have ONE node1 and ONE node2. If multiple node1/node2 are implied, create multiple edges.
    4. RELATIONS: If you find relation that connects existing entity with an entity that not in list, add new node for this relations: extract AS MANE RELATIONS as possible. Be VERY precise about it.
    5. REASONING: First explain your extraction logic, then output JSON.
    
    STRUCTURES DESCRIPTIONS:
      1. Entities (nodes). For each entity, output an object with the following fields:
        - "name": entity name (designation). Copy this information from input.
        - "type": copy this type from input.
        - "base_description": additional information and descriptions about the entity. This field answer the question "What this entity is and how it can be described?" as fully as possible, but based ONLY on an available information. You can copy here all words or sentences that describe this entity in the input text, and combine them with existing description if any.
        - "base_attributes": dictionary of attributes; attributes are some characteristics of the entity that can describe it. For example, if there is an entity chair, and this chair is wooden, there will be attribute "material" : "wood". IMPORTANT: for entities of type "event" attribute "time" is indispensable: it is a string describing time of an event, answers the question "when did this event take place?" ("in the evening", "1042 b.c", "in the Age of the Dragon", etc.). ONLY if time cannot be extracted, this string may be empty: "".
      2. Relations (edges). For every relation that you extract from the text fragment:
        - "node1": name of the first entity. Be careful: DO NOT produce None in this fields, add an entity if it is needed here. Answers the question "Who or what has a connection with another entity?".
        - "node2": name of the second entity. Be careful: DO NOT produce None in this fields, add an entity if it is needed here. Answers the question "Who or what has a connection with node1 entity?".
        - "relation_from1to2": lowercase verb or short phrase, describing relation between node1 and node2. MUST NOT be null or None or Empty. Answers the question "How the FIRST entity connected to the SECOND entity?"
        - "relation_from2to1": lowercase verb or short phrase, describing inverted relation between nodes: from node2 to node1. MUST NOT be null or None or Empty. Answers the question "How the SECOND entity connected to the FIRST entity?". For example, if A "holds" B, then B "is held by" A.
        - "description": additional information, detailed description for relation, describing this connection as fully as possible.
        - "weight": float (default 1.0). Answers the question "How strong are these two entities connected by this relation?". For example, two characters can be friends with weight 1.0 - best friends, and friends with weight 0.3 - almost do not friends, only familiar to each other people.

    OUTPUT:
    <reasoning> 
        Briefly explain extraction.
    </reasoing>
    {{
        "nodes": [
            {{
                "name": "copied from input",
                "type": "copied from input",
                "base_description": "combined: old + new info from text",
                "base_attributes": {{"key": "value"}}
            }}
        ],
        "edges": [
            {{
                "node1": "entity name",
                "node2": "entity name",
                "relation_from1to2": "lowercase verb phrase",
                "relation_from2to1": "lowercase inverse verb phrase",
                "description": "context",
                "weight": "float describing strength or reliability of relation"
            }}
        ]
    }}

    ### EXAMPLE (DO NOT USE IT IN OUTPUT) ###
    Input Text: "In summer 1670, Alice entered the dark forest."
    Input Nodes: {{
            {{"name": "Alice", "type": "character", "base_description": "A girl living in her parents' house", "base_attributes": {{ "age": "12 year old" }} }},
            {{"name": "Dark Forest", "type": "location", "base_description": "A forest located near Alice's parents' house", "base_attributes": {{}} }},
            {{"name": "Alice enters forest", "type": "event", "base_description": "", "base_attributes": {{}} }}
    }}

    <reasoning>
    For character Alice found new description and new attributes, found relations between nodes, for location Dark Forest found new description, also found information for event of entering. Time is "summer 1670".
    </reasoning>
    {{
        "nodes": [
            {{"name": "Alice", "type": "character", "base_description": "A girl living in her parents' house. Enters forest.", "base_attributes": {{}}}},
            {{"name": "Dark Forest", "type": "location", "base_description": "A forest located near Alice's parents' house. Forest which Alice entered", "base_attributes": {{}}}},
            {{"name": "Alice enters forest", "type": "event", "base_description": "Alice enters the forest", "base_attributes": {{"time": "summer 1670"}}}}
        ],
        "edges": [
            {{"node1": "Alice enters forest", "node2": "Alice", "relation_from1to2": "involves", "relation_from2to1": "participates in", "description": "Alice participates in event", "weight": 1.0}},
            {{"node1": "Alice enters forest", "node2": "Dark Forest", "relation_from1to2": "occurs in", "relation_from2to1": "contains event", "description": "Event occurs in forest", "weight": 1.0}}
        ]
    }}
    ### END EXAMPLE ###

    CRITICAL: Output ONLY <reasoning> block followed by JSON. No other text. Preserve all input information.
"""


SYSTEM_PROMPT_ENTITIES_RU = """
    Извлеки сущности и отношения из фрагмента текста.
    ВХОДНЫЕ ДАННЫЕ:
        - Фрагмент текста
        - Кластеры кореференции (группы упоминаний для одной сущности)

    ПРАВИЛА:
        1. Извлеки все значимые сущности. Разреши местоимения с помощью кореференции. Если существуют сущности, не упомянутые в кластерах кореференции, но присутствующие в тексте, извлеки их тоже.
        2. СОБЫТИЯ: извлекай только если они влияют на игровой мир или имеют имя.
        3. ИМЕНА В СКОБКАХ: "Принцесса (Элли)" → name="Элли", type="character".
        4. ИМЯ: краткая каноническая форма, без описаний (например, "Элли", а не "Элли-принцесса").
        5. ОТНОШЕНИЯ: События должны быть связаны между собой отношениями последовательности СТРОГО с realtion следует за/предшествует (ОБЯЗАТЕЛЬНО добавляй эти отношения между событиями в дополнение к другим отношениям для определения последовательности)
    
    СТРУКТУРЫ И ПРАВИЛА:
      1. Сущности (узлы). Для каждой сущности, идентифицированной из кластера кореференции или найденной самостоятельно, выведи объект со следующими полями:
        - "name": имя сущности (обозначение). Имя — это личное имя, без какой-либо дополнительной информации и описаний (даже если оно включено в кластеры кореференции: выбери то, которое без дополнительной информации, самое короткое).
        - "type": один из разрешённых типов, описанных ниже. Убедись, что живые существа — это персонажи!
        - "base_description": дополнительная информация и описания о сущности. Это поле отвечает на вопрос "Что это за сущность и как её можно описать?" максимально полно, но ТОЛЬКО на основе доступной информации. Ты можешь скопировать сюда все слова или предложения, которые описывают эту сущность во входном тексте.
        - "base_attributes": словарь атрибутов; атрибуты — это некоторые характеристики сущности, которые могут её описывать. Например, если есть сущность стул, и этот стул деревянный, то будет атрибут "material" : "wood". 
        ВАЖНО: для сущностей типа "event" атрибут "time" обязателен: это строка, описывающая время события, отвечающая на вопрос "когда произошло это событие?" ("вечером", "1042 г. до н.э.", "в Эпоху Дракона" и т.д.). ТОЛЬКО если время не может быть извлечено, эта строка может быть пустой: "".
      2. Отношения (рёбра). Для каждого найденного отношения:
        - "node1": имя первой сущности. Будь внимателен: НЕ создавай None в этих полях, добавь сущность, если это необходимо. Отвечает на вопрос "Кто или что имеет связь с другой сущностью?".
        - "node2": имя второй сущности. Будь внимателен: НЕ создавай None в этих полях, добавь сущность, если это необходимо. Отвечает на вопрос "Кто или что имеет связь с сущностью node1?".
        - "relation_from1to2": глагол или короткая фраза в нижнем регистре, описывающая отношение между node1 и node2. НЕ ДОЛЖЕН быть null, None или пустым. Отвечает на вопрос "Как ПЕРВАЯ сущность связана со ВТОРОЙ сущностью?"
        - "relation_from2to1": глагол или короткая фраза в нижнем регистре, описывающая обратное отношение между узлами: от node2 к node1. НЕ ДОЛЖЕН быть null, None или пустым. Отвечает на вопрос "Как ВТОРАЯ сущность связана с ПЕРВОЙ сущностью?". Например, если A "держит" B, то B "удерживается A".
        - "description": дополнительная информация, детальное описание отношения, описывающее эту связь как можно полнее.
        - "weight": float (по умолчанию 1.0). Отвечает на вопрос "Насколько сильно эти две сущности связаны этим отношением?". Например, два персонажа могут быть друзьями с весом 1.0 — лучшие друзья, и друзьями с весом 0.3 — почти не друзья, только знакомые друг другу люди.
        ВАЖНО: Отношения всегда должны иметь ОДИН node1 и ОДИН node2. Если подразумевается несколько node1/node2, создай несколько рёбер.
    
    ТИПЫ СУЩНОСТЕЙ. Используй следующие типы для поля "type":
      1. "character" — разумное существо или индивид, действующий в повествовании. Может иметь разные отношения.
      2. "group" — группа персонажей, действующих как единое целое. Эти сущности могут иметь отношения "находится в", "принимает участие в" (событии), "содержит" (персонажа, и персонаж "является частью") и другие различные связи с другими типами и между узлами этого типа.
      3. "location" — географическое или пространственное место. Между сущностями этого типа должны быть рёбра, описывающие пространственные отношения, такие как "связан с", "расположен к северу/югу/востоку/западу от", "имеет дорогу к" и т.д.
      4. "environment_element" — часть или особенность локации. ОБЯЗАТЕЛЬНО должно иметь отношение "находится в", которое связывает его с определённой локацией, где расположен этот элемент.
      5. "item" — физический объект, которым можно владеть или с которым можно взаимодействовать. Может иметь разные отношения.
      6. "event" — действие, происшествие или изменение состояния. События формируют базовую хронологическую и причинно-следственную структуру (фабулу), и они должны иметь поле "time" в "base_attributes".
      ВАЖНО ДЛЯ СОБЫТИЙ: старайся извлечь как можно больше информации о хронологическом порядке событий: между сущностями этого типа ОБЯЗАТЕЛЬНЫ хронологические отношения, такие как "предшествует" и "следует за".

    ФОРМАТ ВЫВОДА:
    <reasoning> 
        Кратко объясни, что ты извлёк и почему. Не более 80 слов.
    </reasoning>
    {{
        "nodes": [
            {{
                "name": "краткое каноническое имя",
                "type": "character | group | location | environment_element | item | event",
                "base_description": "что это за сущность, только на основе текста",
                "base_attributes": {{"ключ": "значение"}}
            }}
        ],
        "edges": [
            {{
                "node1": "имя сущности",
                "node2": "имя сущности",
                "relation_from1to2": "глагольная фраза в нижнем регистре",
                "relation_from2to1": "обратная глагольная фраза в нижнем регистре",
                "description": "контекст связи: что она означает",
                "weight": 1.0
            }}
        ]
    }}

    ### ПРИМЕР (НЕ ИСПОЛЬЗУЙ ЕГО В ВЫВОДЕ) ###
    Входной текст: "Летом 1670 года Алиса вошла в тёмный лес."
    Кластеры кореференции: [["Алиса", "она"], ["тёмный лес", "лес"]]

    <reasoning>
    Найден персонаж Алиса, локация Тёмный лес и событие входа. Время — "лето 1670 года".
    </reasoning>
    {{
        "nodes": [
            {{"name": "Алиса", "type": "character", "base_description": "Девушка", "base_attributes": {{}} }},
            {{"name": "Тёмный лес", "type": "location", "base_description": "Лес", "base_attributes": {{}} }},
            {{"name": "Алиса входит в лес", "type": "event", "base_description": "Алиса входит в лес", "base_attributes": {{"time": "лето 1670 года"}} }}
        ],
        "edges": [
            {{"node1": "Алиса входит в лес", "node2": "Алиса", "relation_from1to2": "включает", "relation_from2to1": "участвует в", "description": "Алиса участвует в событии", "weight": 1.0}},
            {{"node1": "Алиса входит в лес", "node2": "Тёмный лес", "relation_from1to2": "происходит в", "relation_from2to1": "содержит событие", "description": "Событие происходит в лесу", "weight": 1.0}}
        ]
    }}
    ### КОНЕЦ ПРИМЕРА ###

    КРИТИЧНО:
        - Выводи ТОЛЬКО блок <reasoning>, за которым следует JSON. Никакого другого текста.
        - Для событий: "time" в base_attributes обязателен.
        - У отношений должны быть оба направления, непустые строки.
"""

SYSTEM_PROMPT_MERGING_RU = """
    Определи, представляют ли два узла ОДНУ и ту же реальную сущность.
    ВХОДНЫЕ ДАННЫЕ: Два объекта узла с полями name, base_description, base_attributes.

    ПРАВИЛА ПРИНЯТИЯ РЕШЕНИЙ:
        1. ОБЪЕДИНИТЬ, если: одинаковое имя ИЛИ одинаковое описание ИЛИ явно одна и та же сущность в контексте. Ответь на вопрос: "Являются ли эти имена/описания синонимами или относятся к одной сущности?"
        2. НЕ ОБЪЕДИНЯТЬ, если: похожий тип, но разные экземпляры (например, "меч" против "меча короля" без подтверждения).
        3. При объединении: объедини описания и атрибуты. Предпочитай более конкретные значения.
        4. ИМЯ В ОПИСАНИИ: Если A.name появляется в B.base_description → очень вероятно, что одна и та же сущность → ОБЪЕДИНИТЬ.

    ФОРМАТ ВЫВОДА:
    <reasoning>
        Кратко объясни, почему объединить или нет.
    </reasoning>
    {{
        "name": "лучшее выбранное имя",
        "base_description": "объединённое описание",
        "base_attributes": {{"ключ": "значение"}}
    }}

    Если не объединять → все поля пустые: {{"name": "", "base_description": "", "base_attributes": {{}} }}
    КРИТИЧНО: Выводи ТОЛЬКО блок <reasoning>, за которым следует JSON. Никакого другого текста.
"""

SYSTEM_PROMPT_EVENTS_IMPACTS_RU = """
    Определи, как СОБЫТИЯ изменяют СУЩНОСТИ и ОТНОШЕНИЯ.
    ВХОД:
        - Фрагмент текста, в котором упоминаются события (одно или несколько)
        - Названия событий, упомянутых в текстовом фрагменте
        - Список сущностей (с ID), упомянутых в тексте
        - Список отношений (с ID), упомянутых в тексте

    ПРАВИЛА:
        1. Сосредоточься только на ИЗМЕНЕНИЯХ СОСТОЯНИЙ. Для каждого упомянутого события задавай вопрос: "До события: сущность была X. После события: она всё ещё X?"
        2. Для затронутых вершин: предоставь полное новое описание (убедись, что это описание максимально полно описывает сущность для данного состояния, включая факты, которые были известны ранее) и скопируй базовые атрибуты с изменениями для периода ПОСЛЕ события.
        3. Если у сущности НЕТ СОСТОЯНИЙ, и только в этом случае, создай состояние ДО события: ты должна ПОКАЗАТЬ ИЗМЕНЕНИЯ в этой сущности. В этом случае используй структуру "before EVENT_NAME" для названия события в выходном массиве. Если ты создаёшь предыдущее состояние и результирующее состояние, предыдущее состояние ОБЯЗАТЕЛЬНО должно иметь time_end_event, равный текущему событию.
        4. Если в состояниях вершины ты видишь состояние без time_end_event, и одно из событий, с которыми ты сейчас работаешь, является следующим состоянием после открытого состояния (без time_end_event), добавь это открытое состояние в changed_states и запиши для него time_end_event. Только последнее состояние вершины может быть открытым. Не добавляй состояние "before", если для этой сущности уже существуют состояния: предпочитай изменять состояния с time_end_event.
        5. Для затронутых рёбер: отмечай time_start_event (отношение начинается) или time_end_event (отношение заканчивается). НЕ ЗАБУДЬ О ТОМ, ЧТО РЕБРА ТОЖЕ ПОД ВЛИЯНИЕМ СОБЫТИЙ.
        6. Используй ТОЧНЫЕ ID из входных списков. Не придумывай новые ID.
        7. Если ничего не изменяется → возвращай пустые списки.
    Если ты добавила событие BEFORE, обязательно добавь и само событие тоже.

    ВЫХОД:
    <reasoning>
        Для каждого события очень кратко объясни его влияние на вершины и сущности, суммарно менее 100 слов. Ответь на вопрос, нужно ли добавлять состояние before-event. Делай этот блок максимально коротким.
    </reasoning>
    {{
        "events_with_impact": [
            {{
                "event_name": "название события из списка" или "before название события из списка",
                "changed_states": [
                    {{
                        "node_id" : "ДОЛЖЕН совпадать с ID сущности из входных данных",
                        "sid" : "ID состояния, СКОПИРОВАННЫЙ БЕЗ ИЗМЕНЕНИЙ из входного состояния, которое ты хочешь изменить",
                        "time_end_event": "string"
                    }}
                ]
                "affected_nodes": [
                    {{
                        "id": "ДОЛЖЕН совпадать с ID сущности из входных данных",
                        "name": "string",
                        "new_current_description": "полное описание после события или до события",
                        "new_current_attributes": {{"key": "value"}},
                        "time_start_event": "string или null",
                        "time_end_event": "string или null"
                    }}
                ],
                "affected_edges": [
                    {{
                        "id": "ДОЛЖЕН совпадать с ID ребра из входных данных",
                        "new_description": "string",
                        "time_start_event": "string или null",
                        "time_end_event": "string или null"
                    }}
                ]
            }}
        ]
    }}

    ### ПРИМЕР (НЕ ИСПОЛЬЗУЙ ЕГО В ВЫХОДЕ) ###
        Вход:
        Text: "Король умер. Его сын стал новым правителем."
        Events: ["Король умер"]
        Entities: [
            {{"id": "char_king", "name": "Король", "type": "character", "states": 
                {{
                    "sid": "king_becomes_father_king",
                    "current_description": "Король королевства. Теперь у него есть сын.",
                    "current_attributes": {{ "status": "alive", "family_state" : "father" }},
                    "time_start_event": "king_becomes_father",
                    "time_end_event": null,
                }}
            }}, 
            {{"id": "char_son", "name": "Принц", "type": "character"}}
        ]
        Relations: [
            {{"id": "edge_01", "source": "char_son", "target": "char_king", "relation": "является сыном"}}
        ]

        Выход:
        <reasoning>
            Состояние короля изменилось с живого на мёртвого. Состояние сына изменилось с принца на правителя. Отношения не изменились. У сущности Принц нет состояний, поэтому я должна добавить структуру "before EVENT_NAME".
        </reasoning>
        {{
            "events_with_impact": [
                {{
                    "event_name": "before Король умер",
                    "changed_states" : [],
                    "affected_nodes": [
                        {{
                            "id": "char_king",
                            "name": "Король",
                            "new_current_description": "Король королевства",
                            "new_current_attributes": {{ "status": "alive" }},
                            "time_start_event": null,
                            "time_end_event": "Король умер"
                        }},
                        {{
                            "id": "char_son",
                            "name": "Принц",
                            "new_current_description": "Сын правителя королевства",
                            "new_current_attributes": {{ "title": "prince" }},
                            "time_start_event": null,
                            "time_end_event": "Король умер"
                        }}
                    ],
                    "affected_edges": []
                }},
                {{
                    "event_name": "Король умер",
                    "changed_states" : [
                        {{
                            "node_id": "char_king",
                            "sid": "king_becomes_father_king",
                            "time_end_event": "Король умер"
                        }}
                    ],
                    "affected_nodes": [
                        {{
                            "id": "char_king",
                            "name": "Король",
                            "new_current_description": "Умерший король",
                            "new_current_attributes": {{ "status": "dead" }},
                            "time_start_event": "Король умер",
                            "time_end_event": null
                        }},
                        {{
                            "id": "char_son",
                            "name": "Принц",
                            "new_current_description": "Новый правитель королевства",
                            "new_current_attributes": {{ "title": "king" }},
                            "time_start_event": "Король умер",
                            "time_end_event": null
                        }}
                    ],
                    "affected_edges": []
                }}
            ]
        }}
        ### КОНЕЦ ПРИМЕРА ###

    CRITICAL: Выводи ТОЛЬКО блок <reasoning>, после которого идёт JSON. Никакого другого текста. Используй точные ID из входных данных.
"""

SYSTEM_PROMPT_GRAPH_COMPLETION_RU = """
    Дополни нарративный граф знаний, найдя пропущенные сущности и отношения. Старайся найти более 6-8 дополнительных отношений.
    ОСНОВНОЙ ФОКУС: МАКСИМИЗАЦИЯ СВЯЗНОСТИ ЧЕРЕЗ ОТНОШЕНИЯ. Твоя главная цель — восстановить пропущенные рёбра. Извлекай сущности ТОЛЬКО если они строго необходимы для формирования валидных, подтверждённых текстом отношений.

    ВХОДНЫЕ ДАННЫЕ:
        - Фрагмент текста
        - Существующие сущности
        - Существующие отношения

    ПРАВИЛА:
    1. СНАЧАЛА ПРОВЕРЬ: Сравни с существующими списками. НЕ дублируй.
    2. ИЗВЛЕЧЕНИЕ ЧЕРЕЗ ОТНОШЕНИЯ:
        - Просканируй текст на наличие связей: пространственные, владения, участия, причинно-следственные, временные, социальные, изменения состояния.
        - Если связь найдена, извлеки её и обратную связь. Создай сущность только если её ещё не существует.
        - НИКОГДА не оставляй извлечённую сущность изолированной. Каждая новая сущность должна соединяться хотя бы с одним существующим узлом.
    3. СВЯЗАННОСТЬ СОБЫТИЙ:
        - События ОБЯЗАТЕЛЬНО должны быть связаны с: участниками, локациями, причинами, следствиями и ВРЕМЕННЫМ ПОРЯДКОМ (СТРОГО предшествует/следует за).
    4. ФОРМАТИРОВАНИЕ:
        Для missing_entities:
            - name: краткое каноническое имя (например, "кошка", а не "чёрная кошка")
            - type: один из: character, group, location, environment_element, item, event
            - base_description: что это за сущность (1-2 предложения)
            - base_attributes: словарь характеристик (например, {{"цвет": "черный", "материал": "дерево"}})
            - reason: почему эта сущность была пропущена и доказательство из текста
            - chunk_reference: точная цитата из текста, упоминающая эту сущность

        Для missing_relations:
            - node1: имя первой сущности (должно совпадать с существующим или новым именем сущности)
            - node2: имя второй сущности (должно совпадать с существующим или новым именем сущности)
            - relation_from1to2: глагольная фраза в нижнем регистре (например, "находится на", "владеет", "участвует в")
            - relation_from2to1: обратная глагольная фраза в нижнем регистре (например, "имеет на себе", "находится во владении", "включает")
            - description: контекст этой связи
            - weight: float 0.0-1.0 (1.0 = сильное/явное, 0.5 = подразумеваемое)
            - reason: почему это отношение было пропущено и доказательство из текста
            - chunk_reference: точная цитата из текста, подразумевающая это отношение

    5. ДОКАЗАТЕЛЬСТВА (ОБЯЗАТЕЛЬНО):
        - reason: объясни, почему это было пропущено и как текст это подтверждает.
        - chunk_reference: ПРЯМАЯ ЦИТАТА из текста. БЕЗ ПЕРЕСКАЗА. Если точная цитата длинная, обрежь её с "...".

    ФОРМАТ ВЫВОДА:
    <reasoning>
        Кратко опиши пропущенные сущности, восстановленные отношения и ключевые улучшения связности. Менее 80 слов.
    </reasoning>
    {{
        "missing_entities": [
            {{
                "name": "краткое каноническое имя",
                "type": "character | group | location | environment_element | item | event",
                "base_description": "1-2 предложения, описывающие эту сущность, могут быть из текста",
                "base_attributes": {{"ключ": "значение"}},
                "reason": "строка",
                "chunk_reference": "точная цитата"
            }}
        ],
        "missing_relations": [
            {{
                "node1": "имя сущности",
                "node2": "имя сущности",
                "relation_from1to2": "глагол в нижнем регистре",
                "relation_from2to1": "обратный глагол в нижнем регистре",
                "description": "контекст связи",
                "weight": "число с плавающей запятой от 0.00 до 1.00",
                "reason": "строка",
                "chunk_reference": "точная цитата"
            }}
        ]
    }}

    ### ПРИМЕР (НЕ ИСПОЛЬЗУЙ ЕГО В ВЫВОДЕ) ###
    Входные данные:
    Текст: "Чёрная кошка спала на деревянной кровати в спальне. Мэри наблюдала за своим питомцем из дверного проёма."
    Сущности: [
        {{"name": "кровать", "type": "item", "description": "Деревянная кровать"}},
        {{"name": "спальня", "type": "location", "description": "Комната"}}
    ]
    Существующие отношения: []

    Вывод:
    <reasoning>
        Найдено 2 пропущенные сущности: кошка (персонаж) и Мэри (персонаж). Найдено 3 пропущенных отношения: кошка-на-кровати, кровать-в-спальне, Мэри-наблюдает-за-кошкой. Кошка была упомянута, но не извлечена. Мэри была упомянута по имени, но не извлечена. Пространственные отношения не были зафиксированы.
    </reasoning>
    {{
        "missing_entities": [
            {{
                "name": "кошка",
                "type": "character",
                "base_description": "Чёрная кошка, питомец Мэри",
                "base_attributes": {{"цвет": "черный", "хозяйка": "Мэри"}},
                "reason": "Кошка — разумное существо (персонаж), упомянутое в тексте, но не извлечённое",
                "chunk_reference": "Чёрная кошка спала на деревянной кровати"
            }},
            {{
                "name": "Мэри",
                "type": "character",
                "base_description": "Человек, который владеет кошкой и наблюдает за ней",
                "base_attributes": {{"роль": "владелец питомца"}},
                "reason": "Мэри — персонаж, упомянутый по имени, но не извлечённый",
                "chunk_reference": "Мэри наблюдала за своим питомцем из дверного проёма"
            }}
        ],
        "missing_relations": [
            {{
                "node1": "кошка",
                "node2": "кровать",
                "relation_from1to2": "находится на",
                "relation_from2to1": "имеет на себе",
                "description": "Кошка спит на кровати",
                "weight": 1.0,
                "reason": "Явное пространственное отношение 'на' не было извлечено",
                "chunk_reference": "кошка спала на деревянной кровати"
            }},
            {{
                "node1": "кровать",
                "node2": "спальня",
                "relation_from1to2": "находится в",
                "relation_from2to1": "содержит",
                "description": "Кровать находится в спальне",
                "weight": 1.0,
                "reason": "Явное пространственное отношение 'в' не было извлечено",
                "chunk_reference": "кровать в спальне"
            }},
            {{
                "node1": "Мэри",
                "node2": "кошка",
                "relation_from1to2": "владеет",
                "relation_from2to1": "находится во владении",
                "description": "Мэри владеет кошкой как питомцем",
                "weight": 1.0,
                "reason": "Отношение владения, подразумеваемое фразой 'её питомец', не было извлечено",
                "chunk_reference": "Мэри наблюдала за своим питомцем"
            }}
        ]
    }}
    ### КОНЕЦ ПРИМЕРА ###

    КРИТИЧНО:
        - Выводи ТОЛЬКО блок <reasoning>, за которым следует JSON. Никакого другого текста.
        - Каждый новый элемент ОБЯЗАТЕЛЬНО должен содержать chunk_reference.
        - Отдавай приоритет отношениям над сущностями. Изолированные сущности недопустимы.
"""

SYSTEM_PROMPT_ENTITIES_NAMES_RU = """
    Ты должен точно извлечь все сущности из фрагмента текста и не пропустить ни одной сущности.
    ЗАДАЧА: Извлечь имена и типы сущностей из фрагмента текста.

    ВХОДНЫЕ ДАННЫЕ:
        - Фрагмент текста
        - Кластеры кореференции

    ПРАВИЛА:
        1. Извлеки все значимые сущности. Разреши местоимения с помощью кореференции. Если существуют сущности, не упомянутые в кластерах кореференции, но присутствующие в тексте, извлеки их тоже — НЕ ПОЛАГАЙСЯ ТОЛЬКО НА КЛАСТЕРЫ КОРЕФЕРЕНЦИИ.
        2. СОБЫТИЯ: извлекай только если они влияют на игровой мир или имеют имя.
        3. ИМЕНА В СКОБКАХ: "Принцесса (Элли)" → name="Элли", type="character".
        4. ИМЯ: краткая каноническая форма, без описаний (например, "Элли", а не "Элли-принцесса").

    ТИПЫ СУЩНОСТЕЙ (используй именно эти типы):
        1. "character" — разумное существо или индивид, действующий в повествовании.
        2. "group" — группа персонажей, действующих как единое целое.
        3. "location" — географическое или пространственное место.
        4. "environment_element" — часть или особенность локации. Должен иметь атрибут "находится в".
        5. "item" — физический объект, которым можно владеть или с которым можно взаимодействовать.
        6. "event" — действие, происшествие или изменение состояния. События формируют базовую хронологическую и причинно-следственную структуру (фабулу).

    ФОРМАТ ВЫВОДА:
    <reasoning>
        Кратко объясни извлечение, для каждого события коротко (максимум три слова) опиши, нужно ли его извлекать.
    </reasoning>
    {{
        "nodes": [
            {{
                "name": "краткое каноническое имя",
                "type": "character | group | location | environment_element | item | event"
            }}
        ]
    }}

    ### ПРИМЕР (НЕ ИСПОЛЬЗУЙ ЕГО В ВЫВОДЕ) ###
    Входной текст: "Летом 1670 года Алиса вошла в тёмный лес."
    Кластеры кореференции: [["Алиса", "она"], ["тёмный лес", "лес"]]

    <reasoning>
        Найден персонаж Алиса, локация Тёмный лес и событие входа.
    </reasoning>
    {{
        "nodes": [
            {{ "name": "Алиса", "type": "character" }},
            {{ "name": "Тёмный лес", "type": "location" }},
            {{ "name": "Алиса входит в лес", "type": "event" }}
        ]
    }}
    ### КОНЕЦ ПРИМЕРА ###

    КРИТИЧНО: Выводи ТОЛЬКО блок <reasoning>, за которым следует JSON. Никакого другого текста.
"""

SYSTEM_PROMPT_MERGING_NAMES_RU = """
    Определи, относятся ли два имени сущности к ОДНОЙ и той же сущности.
    ВХОДНЫЕ ДАННЫЕ: два имени и их контексты.
    ПРАВИЛА:
        1. ОБЪЕДИНИТЬ, если: имена являются синонимами или явно относятся к одной сущности в контексте. Ответь на вопрос: "Описывают ли эти имена одну сущность?"
        2. НЕ ОБЪЕДИНЯТЬ, если: контексты указывают на разные сущности.
        3. При объединении: создай одно лучшее имя.

    ФОРМАТ ВЫВОДА:
        <reasoning>
            Объяснение решения, почему узлы похожи или почему они различны.
        </reasoning>
        {{ "name": "лучшее выбранное имя" }}
        
        Если не объединять: {{ "name": "" }}

    КРИТИЧНО: Выводи ТОЛЬКО блок <reasoning>, за которым следует JSON. Никакого другого текста.
"""

SYSTEM_PROMPT_ENTITIES_WITH_NAMES_RU = """
    Обогати предварительно извлечённые сущности информацией из текста и извлеки отношения между ними (как можно больше отношений).
    ВХОДНЫЕ ДАННЫЕ:
        - Фрагмент текста
        - Список сущностей, упомянутых в этом фрагменте текста, с именами, типами и base_description, которые уже были извлечены.
    ПРАВИЛА:
    1. СУЩНОСТИ: Заполни все обязательные поля для всех сущностей, упомянутых во входных данных. Если есть base_description и base_attributes, объедини их с информацией, извлечённой из текущего текста. Старайся не потерять никакую информацию.
    2. ОТНОШЕНИЯ: Извлеки ОБА направления для каждой связи:
        - relation_from1to2: Как node1 связан с node2 (например, "держит")
        - relation_from2to1: Обратное отношение (например, "удерживается")
    3. ОТНОШЕНИЯ: Отношения всегда должны иметь ОДИН node1 и ОДИН node2. Если подразумевается несколько node1/node2, создай несколько рёбер.
    4. ОТНОШЕНИЯ: Если ты находишь отношение, которое связывает существующую сущность с сущностью, отсутствующей в списке, добавь новый узел для этого отношения: извлекай КАК МОЖНО БОЛЬШЕ ОТНОШЕНИЙ. Будь ОЧЕНЬ точен в этом.
    5. ОТНОШЕНИЯ: События должны быть связаны между собой отношениями последовательности СТРОГО с realtion следует за/предшествует (ОБЯЗАТЕЛЬНО добавляй эти отношения между событиями в дополнение к другим отношениям для определения последовательности)
    6. РАССУЖДЕНИЕ: Сначала объясни свою логику извлечения, затем выведи JSON.
    
    ОПИСАНИЯ СТРУКТУР:
      1. Сущности (узлы). Для каждой сущности выведи объект со следующими полями:
        - "name": имя сущности (обозначение). Скопируй эту информацию из входных данных.
        - "type": скопируй этот тип из входных данных.
        - "base_description": дополнительная информация и описания о сущности. Это поле отвечает на вопрос "Что это за сущность и как её можно описать?" максимально полно, но ТОЛЬКО на основе доступной информации. Ты можешь скопировать сюда все слова или предложения, которые описывают эту сущность во входном тексте, и объединить их с существующим описанием, если оно есть.
        - "base_attributes": словарь атрибутов; атрибуты — это некоторые характеристики сущности, которые могут её описывать. Например, если есть сущность стул, и этот стул деревянный, то будет атрибут "material" : "wood". ВАЖНО: для сущностей типа "event" атрибут "time" обязателен: это строка, описывающая время события, отвечающая на вопрос "когда произошло это событие?" ("вечером", "1042 г. до н.э.", "в Эпоху Дракона" и т.д.). ТОЛЬКО если время не может быть извлечено, эта строка может быть пустой: "".
      2. Отношения (рёбра). Для каждого отношения, которое ты извлекаешь из фрагмента текста:
        - "node1": имя первой сущности. Будь внимателен: НЕ создавай None в этих полях, добавь сущность, если это необходимо. Отвечает на вопрос "Кто или что имеет связь с другой сущностью?".
        - "node2": имя второй сущности. Будь внимателен: НЕ создавай None в этих полях, добавь сущность, если это необходимо. Отвечает на вопрос "Кто или что имеет связь с сущностью node1?".
        - "relation_from1to2": глагол или короткая фраза в нижнем регистре, описывающая отношение между node1 и node2. НЕ ДОЛЖЕН быть null, None или пустым. Отвечает на вопрос "Как ПЕРВАЯ сущность связана со ВТОРОЙ сущностью?"
        - "relation_from2to1": глагол или короткая фраза в нижнем регистре, описывающая обратное отношение между узлами: от node2 к node1. НЕ ДОЛЖЕН быть null, None или пустым. Отвечает на вопрос "Как ВТОРАЯ сущность связана с ПЕРВОЙ сущностью?". Например, если A "держит" B, то B "удерживается A".
        - "description": дополнительная информация, детальное описание отношения, описывающее эту связь как можно полнее.
        - "weight": float (по умолчанию 1.0). Отвечает на вопрос "Насколько сильно эти две сущности связаны этим отношением?". Например, два персонажа могут быть друзьями с весом 1.0 — лучшие друзья, и друзьями с весом 0.3 — почти не друзья, только знакомые друг другу люди.

    ФОРМАТ ВЫВОДА:
    <reasoning>
        Кратко объясни извлечение.
    </reasoning>
    {{
        "nodes": [
            {{
                "name": "скопировано из входных данных",
                "type": "скопировано из входных данных",
                "base_description": "объединённое: старая + новая информация из текста",
                "base_attributes": {{ "ключ": "значение" }}
            }}
        ],
        "edges": [
            {{
                "node1": "имя сущности",
                "node2": "имя сущности",
                "relation_from1to2": "глагольная фраза в нижнем регистре",
                "relation_from2to1": "обратная глагольная фраза в нижнем регистре",
                "description": "контекст",
                "weight": "float, описывающий силу или надёжность отношения"
            }}
        ]
    }}

    ### ПРИМЕР (НЕ ИСПОЛЬЗУЙ ЕГО В ВЫВОДЕ) ###
    Входной текст: "Летом 1670 года Алиса вошла в тёмный лес."
    Входные узлы: {{
            {{ "name": "Алиса", "type": "character", "base_description": "Девочка, живущая в доме родителей", "base_attributes": {{"age": "12 лет"}} }},
            {{ "name": "Тёмный лес", "type": "location", "base_description": "Лес, расположенный недалеко от дома родителей Алисы", "base_attributes": {{}} }},
            {{ "name": "Алиса входит в лес", "type": "event", "base_description": "", "base_attributes": {{}} }}
    }}

    <reasoning>
        Для персонажа Алиса найдено новое описание и новые атрибуты, найдены отношения между узлами, для локации Тёмный лес найдено новое описание, также найдена информация для события входа. Время — "лето 1670 года".
    </reasoning>
    {{
        "nodes": [
            {{ "name": "Алиса", "type": "character", "base_description": "Девочка, живущая в доме родителей. Входит в лес.", "base_attributes": {{}} }},
            {{ "name": "Тёмный лес", "type": "location", "base_description": "Лес, расположенный недалеко от дома родителей Алисы. Лес, в который вошла Алиса", "base_attributes": {{}} }},
            {{ "name": "Алиса входит в лес", "type": "event", "base_description": "Алиса входит в лес", "base_attributes": {{ "time": "лето 1670" }} }}
        ],
        "edges": [
            {{ "node1": "Алиса входит в лес", "node2": "Алиса", "relation_from1to2": "включает", "relation_from2to1": "участвует в", "description": "Алиса участвует в событии", "weight": 1.0 }},
            {{ "node1": "Алиса входит в лес", "node2": "Тёмный лес", "relation_from1to2": "происходит в", "relation_from2to1": "содержит событие", "description": "Событие происходит в лесу", "weight": 1.0 }}
        ]
    }}
    ### КОНЕЦ ПРИМЕРА ###

    КРИТИЧНО: Выводи ТОЛЬКО блок <reasoning>, за которым следует JSON. Никакого другого текста. Сохрани всю информацию из входных данных.
"""

SYSTEM_PROMPT_MERGING_IN_GRAPH_RU = """
    Объедини две сущности в одну сущность, сохрани как можно больше данных. Создай для двух сущностей объединённое base_description, объедини список атрибутов и список состояний: для состояний будь внимателен:
        1. если два состояния из разных сущностей похожи — создай объединённое состояние.
        2. если два состояния из разных сущностей различны (имеют разное время), ты должен в этом случае просто скопировать состояния в итоговую объединенную сущность.
        3. также будь внимателен с time_start_event и time_end_event: копируй их аккуратно и при объединении обращай внимание на моменты смены состояний (чтобы time_end_event для предыдущего состояния был time_start_event для следующего состояния)

    ВХОДНЫЕ ДАННЫЕ: Два объекта сущностей с полями name, base_description, base_attributes и списком states.

    ФОРМАТ ВЫВОДА:
    <reasoning>
        Объясни, как ты объединишь две сущности. Не более 80 слов.
    </reasoning>
    {{
          "name": "строка (выбери лучшее имя)",
          "type": "строка (один из типов, скопированных из входных узлов)",
          "base_description": "строка (объединённое описание)",
          "base_attributes": {{"ключ": "значение"}},
          "states": [
                {{
                    "sid": "строка (id для состояния, скопированный из входа или созданный, если состояния были объединены)",
                    "current_description": "строка (объединённое описание из двух состояний или скопированное описание)",
                    "current_attributes": {{"ключ": "значение"}},
                    "time_start_event": "строка (id события, скопированный из входа)",
                    "time_end_event": "строка (id события, скопированный из входа)"
                }}
          ]
    }}
    КРИТИЧНО: Выводи ТОЛЬКО блок <reasoning>, за которым следует JSON. Никакого другого текста.
"""