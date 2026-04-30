TEST_TASKS = [
    {
        "id": "test",
        "query": "Write a short story for a future quest (do not structure it now) about Morgiana and her ordinary life.",
        "world_context": """
                The story unfolds in a modest Persian town, grounded in realistic social hierarchies yet touched by a single thread of magic: a hidden cave in the forest, sealed 
                by an enchanted command—“Open, Sesame.” This cave, filled with seemingly endless treasure, belongs to a band of forty ruthless thieves. The world operates on clear 
                moral and practical logic—greed leads to ruin, while humility, cleverness, and loyalty are rewarded. Magic exists, but only as a fixed, rule-bound element; 
                it does not interfere with daily life beyond this one extraordinary location.
                The society depicted includes merchants, laborers, slaves, and craftsmen. Slavery is present but not absolute—exceptional service can lead to freedom and social 
                elevation. Wealth is inherited or acquired through fortune, but its moral value depends entirely on how it is used.
                
                Main characters and their relationships:
                 - Ali Baba is a poor but honest woodcutter. He stumbles upon the cave’s secret not through ambition, but by chance. 
                 He remains modest after gaining wealth and treats others with kindness, including his late brother’s slave, Morgiana. 
                 His role is passive in action but central as a moral anchor.
                 - Cassim, Ali Baba’s elder brother, represents greed and social aspiration. Married to a wealthy woman, he immediately seeks to exploit the cave for personal gain. 
                 His arrogance and forgetfulness lead to his death, cutting his role short but triggering the main conflict.
                 - Morgiana, a female slave in Cassim’s household, is the true agent of the story. Intelligent, observant, and courageous, she acts decisively to protect Ali Baba’s 
                 family. Her relationship with Ali Baba evolves from servitude to familial trust; ultimately, she is freed and married to his son—a rare upward mobility 
                 that underscores the tale’s moral fairness.
                 - The Captain of the Forty Thieves serves as the primary antagonist. He is cunning and relentless, using disguise and deception twice to hunt down 
                 the person who breached his secret. However, he underestimates Morgiana, assuming threats come only from men. His rigid worldview leads to his downfall.
                 - Ali Baba’s son plays a minor but functional role: he unknowingly facilitates the captain’s second infiltration by befriending the disguised robber.
                His presence bridges generations and allows Morgiana’s final act of heroism to unfold in a domestic setting.
                In this world, the conflict isn’t driven by epic battles or divine intervention, but by human traits—curiosity, greed, loyalty, and wit—played out 
                in a setting where a magical secret disrupts ordinary life. The story that follows (Ali Baba discovering the cave, Cassim’s death, and Morgiana thwarting two 
                assassination attempts) is simply the natural consequence of how these characters interact within this morally coherent, semi-realistic world.
            """,
        "expected": """
                The morning was hot and overcast. Everything foretold a sandstorm, and Morgiana wanted to buy provisions before going outside became dangerous. 
                She went to the market: only a few merchants had dared to come today and set out their goods, and even they kept glancing anxiously at the horizon, 
                trying to spot the approaching clouds in time. Morgiana bought everything on her list, bargaining for the best price she could. 
                After that, she went to the butcher’s shop, which was always open—being inside a solid building, the butcher was not worried about the storm, 
                as he could easily protect his goods. Having bought meat there, she visited a friend and gave her a ring and a necklace she had promised long ago. 
                After receiving some fresh gossip, Morgiana returned home and began preparing lunch. Two hours later, the sandstorm began.
            """,
        "metric": ["mauve", "distinct-n", "repetition-n", "world_consistency"],
        "category": "story for a quest"
    }
]

GRAPH_TEST_DATASET = [
    {
        "path": "assets/documents/NOTEBOOK STORY.txt",
        "language": "en",
        "embedding_model_options": {
            "name": "embeddings",
            "option": "hf_local",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "expected_values": {
            "nodes": 11,
            "edges": 42,
            "characters": 4,
            "groups": 1,
            "locations": 1,
            "location_elements": 2,
            "events": 2,
            "items": 1,
            "total_states": 4,
            "nodes_with_gt_2_states": 2,
        }
    },

    {
        "path": "assets/documents/RUSTY LAKE SHORT.txt",
        "language": "en",
        "embedding_model_options": {
            "name": "embeddings",
            "option": "hf_local",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "expected_values": {
            "nodes": 15,
            "edges": 48,
            "characters": 7,
            "groups": 1,
            "locations": 2,
            "location_elements": 0,
            "events": 3,
            "items": 2,
            "total_states": 2,
            "nodes_with_gt_2_states": 1,
        }
    },

    {
        "path": "assets/documents/NUCLEAR CRYPT SHORT.txt",
        "language": "ru",
        "embedding_model_options": {
            "name": "embeddings",
            "option": "hf_local",
            "model_name": "ai-forever/ru-en-RoSBERTa"
        },
        "expected_values": {
            "nodes": 15,
            "edges": 54,
            "characters": 1,
            "groups": 2,
            "locations": 3,
            "location_elements": 0,
            "events": 4,
            "items": 5,
            "total_states": 6,
            "nodes_with_gt_2_states": 3,
        }
    },
]

TEST_DATA_TEXT1 = {
    "path_to_graph": "assets/graphs/elden_ring_lore.json",
    "path_to_text": "assets/documents/ELDEN RING LORE.txt",
    "text_summary": """
        The Greater Will is an Outer God whose power manifests through the Elden Ring and the Erdtree. 
        Queen Marika becomes its vessel and establishes the Golden Order to enforce its will. 
        Before this, dragons ruled the Lands Between but declined after their god abandoned them. 
        With the rise of the Golden Order, beings tied to the primordial “crucible” (like Omens) became persecuted.
        Marika wages wars, including against the Fire Giants, and allies with figures like Hoarah Loux (Godfrey). 
        She and her other self, Radagon, are two aspects of one being. 
        Their children, including Malenia and Miquella, are cursed. Other Outer Gods also influence events.
        Ranni rejects the Greater Will and orchestrates the Night of the Black Knives, killing Godwyn. 
        In response, Marika shatters the Elden Ring, triggering the Shattering war among demigods, each seeking power or change.
        As the Greater Will’s influence weakens, the Tarnished return to the Lands Between, 
        aiming to restore or redefine the world’s order.
    """,
    "language": "en",
    "tasks": [
        {
            "category": "quest",
            "query": "Create a new Elden Ring-style side quest involving a Tarnished helping a minor faction or individual affected by the Shattering. The quest should involve moral ambiguity, hidden truth, and a choice between loyalty to the Golden Order or defiance of it. Do not reuse known canonical events.",
            "reference": """
                A Tarnished encounters a wandering scholar from a minor ruined order who claims to have discovered fragments 
                of forbidden knowledge about the Shattering. The scholar asks for protection while traveling to an ancient site where truth 
                about the Erdtree’s origins may be revealed. Along the journey, it becomes unclear whether the scholar seeks 
                enlightenment or intends to destabilize remaining order. At the destination, the Tarnished must choose to either help seal 
                the knowledge away or release it to the world, risking further chaos.
            """
        },
        {
            "category": "dialogue",
            "query": "Write a short in-game style dialogue where a guide-like spirit offers the player a pact and explains their role in the Lands Between.",
            "reference": """
                Melina: "I offer you an accord. Let me travel with you, and I will guide you to the foot of the Erdtree."
                Tarnished: "Why help me?"
                Melina: "You are Tarnished. You must walk a path to become Elden Lord. I have my own purpose... and I will see it through."
            """
        },
        {
            "category": "character description",
            "query": "Create a new Elden Ring character connected to existing factions or figures, but not mentioned in the source text. The character should be linked to known entities like Ranni or the Golden Order, but must be original. Something about hald-wolves will be interesting, I suppose.",
            "reference": """
                Blaidd is a half-wolf warrior bound to Ranni the Witch by fate and loyalty. He serves as her shadow and protector, 
                tasked by the Two Fingers to watch over her Empyrean destiny. Despite his feral nature, Blaidd is deeply loyal and struggles between instinct and duty. 
                As Ranni rejects the Greater Will, Blaidd becomes entangled in her fate, ultimately unable to escape the influence of the very forces 
                he was created to serve.
            """
        },
        {
            "category": "location description",
            "query": "Create a description for a major capital city in Elden Ring associated with the Golden Order and royal rule.",
            "reference": """
                Leyndell, Royal Capital is the seat of the Erdtree and the center of the Golden Order’s power. 
                It is a vast city built around the massive golden tree, filled with knights, golden architecture, and ancient structures 
                tied to the rule of Queen Marika. During the Shattering, the capital becomes a key stronghold defended by Morgott, who protects 
                it despite being an Omen. The city symbolizes the peak and decay of the Golden Order’s influence.
            """
        },
        {
            "category": "item description",
            "query": "Create a new mystical item tied to divine or forbidden power in the Elden Ring world. It should feel like a fragment of a larger cosmic truth or broken order, but must not directly reference existing Great Runes or known artifacts.",
            "reference": """
                Veilshard of Continuance is a fractured relic said to originate from an unknown layer of the Elden Ring’s structure. 
                When held, it subtly alters perception of time, allowing the bearer to glimpse alternate outcomes of past actions. However, 
                prolonged use erodes memory and identity, as if the world itself resists being understood outside its intended order.
            """
        }
    ]
}

TEST_DATA_TEXT2 = {
    "path_to_text": "assets/documents/Leisure Suit Larry 6.txt",
    "path_to_graph": "assets/graphs/leisure_suit_larry.json",
    "text_summary": """
        Larry Laffer arrives at a luxury spa after losing a dating show, where he is treated poorly as a non-paying guest. 
        Throughout his stay, he helps several women, including Gammie with her weight-loss treatment, Shablee by finding her a dress for a date, 
        and Char with gathering items—though many encounters end in rejection or embarrassment. He also meets Cavaricchi and Burgundy, who invite 
        him to a sauna but exclude him, and Thunderbird, a dominatrix who humiliates him. Larry assists Merrily with her dream of bungee jumping, only 
        to accidentally fall himself. Despite repeated mishaps, Larry persists. In the end, after completing tasks for Shamara, he finally gains her affection, 
        concluding his misadventure on a successful note.
    """,
    "language": "en",
    "tasks": [
        {
            "category": "quest",
            "query": "Design a new comedic adventure puzzle sequence set in the same luxury spa. The quest should involve helping a different eccentric guest achieve a risky personal goal by stealthily acquiring and duplicating a restricted access item. The sequence should culminate in an ironic, highly public mishap for the protagonist while the guest finally attains their desire, maintaining the game's signature blend of slapstick and puzzle-solving.",
            "reference": """
                Larry assists Merrily Lowe at the pool bar by obtaining the diving tower key from the lifeguard, 
                making an impression of it in a bar of soap, filing a copy with a bastard file, and returning it to her. 
                They climb to the bungee platform, where she shares her "Words of Wisdom," but Larry trips over the cords and falls, 
                accidentally broadcasting his naked descent to the entire spa.
            """
        },
        {
            "category": "dialogue",
            "query": "Write a short, in-game style script dialogue where a weary spa performer negotiates a private meeting with the protagonist. The exchange should highlight the performer's cynical exhaustion and the protagonist's hesitant agreement, formatted strictly as `Character: \"line\"` pairs.",
            "reference": """
                Burgundy: "God-dammit, Larry! You got steam room privileges? I'd give a week's pay to get naked and sweat it out now!"
                Larry: "Uh, yeah, I think so..."
                Burgundy: "Good. I'll meet you there as soon as I get out of this dress."
            """
        },
        {
            "category": "character description",
            "query": "Create a description for a new secondary NPC at the same exclusive health resort. The character should be a physically imposing, stern staff member or guest with a commanding, slightly intimidating demeanor. Detail their appearance, their specific request for a restrictive or unconventional accessory, and how they treat the protagonist once their demand is fulfilled, keeping the tone playfully absurd.",
            "reference": """
                Thunderbird is a rough, tough, no-nonsense dominatrix who works out in the spa's weight room. 
                She desires new handcuffs. When given them, she invites Larry to her room, attaches him to a floor lamp with a diamond-studded dog collar, 
                and orders him to crawl and bark, showcasing her dominant physique and commanding personality.
            """
        },
        {
            "category": "location description",
            "query": "Describe a specialized maintenance or treatment room within the same luxury spa facility. The space should feature a prominent, slightly outdated mechanical apparatus that requires hands-on repair. Highlight its clinical yet utilitarian aesthetic, and emphasize how it functions as an interactive puzzle hub where the protagonist must diagnose and fix interconnected mechanical failures using scavenged parts.",
            "reference": """
                The Cellulite Drainage Salon is a clinical room housing a large, complex treatment machine with a dry piston shaft, 
                a leaking vacuum hose, and a clogged filter tank. The protagonist must diagnose these mechanical failures, 
                source replacement parts from other spa areas, and reassemble the system to activate the treatment for the front desk clerk.
            """
        },
        {
            "category": "item description",
            "query": "Describe a mundane, everyday object found in the spa's guest amenities that has been cleverly modified to serve a clandestine purpose. Focus on how a soft, common material was used to capture a precise mechanical pattern from a restricted item, enabling the protagonist to bypass security through traditional adventure game logic.",
            "reference": """
                Impressed Soap is a standard bar of soap that has been carefully pressed with a metal key, 
                leaving a precise negative impression of the key's teeth. It is a crucial intermediate tool for duplicating keys, 
                representing a classic adventure game puzzle mechanic where everyday bathroom items are repurposed for stealthy access.
            """
        }
    ]
}


TEST_DATA_TEXT3 = {
    "path_to_graph": "assets/graphs/inazuma_lore.json",
    "path_to_text": "assets/documents/INAZUMAS MAIN QUEST.txt",
    "text_summary": """
        The Traveler arrives in Inazuma, a nation ruled by the Electro Archon, the Raiden Shogun, 
        who enforces isolation and the Vision Hunt Decree—confiscating Visions from citizens, stripping many of their ambition and purpose. 
        This causes widespread unrest and resistance.

        The Traveler meets allies like Ayaka and Thoma and witnesses the growing oppression. A resistance led by Sangonomiya Kokomi 
        fights the Shogunate, while the Fatui secretly manipulate events to destabilize Inazuma further.

        It is revealed that the true Archon, Raiden Ei, has retreated into the Plane of Euthymia, leaving a puppet Shogun to enforce her ideal 
        of "eternity"—a world without change or loss.

        The Traveler confronts the Shogun and then Ei herself within her inner realm. Through a duel and witnessing 
        the will of Inazuma's people embodied in their Visions, Ei realizes the harm of her ideology.

        She abandons the Vision Hunt Decree, restores the Visions, ends the civil conflict, and begins guiding Inazuma toward a new future.
    """,
    "language": "en",
    "tasks": [
        {
            "category": "quest",
            "query": "Design a new side quest set in Inazuma where the Traveler helps a former Vision bearer cope with the loss of their aspirations after the Vision Hunt Decree. The sequence should involve investigating rumors of a hidden sanctuary where confiscated Visions are kept, navigating moral choices about whether to expose the truth or protect those who still hope, and culminating in a quiet moment of reflection rather than combat.",
            "reference": """
                In the quest "The Meaning of Meaningless Waiting," the Traveler meets Tejima, a former samurai whose Vision was confiscated. 
                Without his Vision, he struggles to remember why he stayed in Konda Village. The Traveler helps him find a letter revealing 
                he waited for a lover who never returned. Tejima chooses to remain, finding new meaning in patience rather than action.
            """
        },
        {
            "category": "dialogue",
            "query": "Write a short in-game style dialogue where Yae Miko speaks with the Traveler about the nature of eternity and human ambition. The exchange should reflect Yae's playful yet profound wisdom, the Traveler's outsider perspective, and hint at the tension between Ei's ideals and mortal desires. Format strictly as `Character: \"line\"` pairs.",
            "reference": """
                Yae Miko: "Eternity is not a destination, little one. It is the space between heartbeats where hope still flickers."
                Traveler: "But if nothing changes, doesn't hope itself fade?"
                Yae Miko: "Ah, but that is precisely why the Shogun fears it. A flicker can become a flame... and flames, as you know, have a way of spreading."
            """
        },
        {
            "category": "character description",
            "query": "Create a description for a new secondary NPC connected to the Watatsumi Resistance or the Yashiro Commission. The character should be a scout, messenger, or local guide with knowledge of Inazuma's hidden paths. Detail their appearance, their role in the ongoing conflict, and a personal motivation that ties them to the broader struggle against the Vision Hunt Decree.",
            "reference": """
                Teppei is a young, earnest soldier of the Watatsumi Resistance, initially assigned to logistics but eager to prove himself on the front lines. 
                He wears patched resistance armor and carries a simple spear. His motivation stems from a desire to protect his homeland and earn recognition, 
                though his enthusiasm sometimes outpaces his experience. He later becomes captain of the special operations unit "Herring I."
            """
        },
        {
            "category": "location description",
            "query": "Describe a secluded shrine or ruined temple on Narukami Island that serves as a secret meeting place for those opposing the Vision Hunt Decree. The space should feature weathered torii gates, lingering Electro residue from confiscated Visions, and a central stone monument tied to ancient Inazuman beliefs about ambition and the heavens. Emphasize its role as a quiet hub for clandestine planning.",
            "reference": """
                The Komore Teahouse is a hidden establishment nestled in the cliffs of Narukami Island, accessible only through a secret passage. 
                Its interior blends traditional Inazuman architecture with subtle resistance symbolism—carved cranes representing freedom, hidden 
                compartments for messages, and a view of the sea that reminds visitors of the world beyond the Sakoku Decree. It serves as the primary safehouse 
                for the Kamisato Clan's covert operations.
            """
        },
        {
            "category": "item description",
            "query": "Describe a newly discovered Inazuman artifact tied to the concept of fleeting ambition and the Electro element. It should feel like a crystallized fragment of a confiscated Vision or a tear from the Sacred Sakura, capable of temporarily resonating with lost desires, but carrying a cost of emotional echo or temporal dissonance.",
            "reference": """
                The Omamori of Unspoken Wishes is a small, silk-wrapped charm imbued with faint Electro energy, 
                originally crafted at the Grand Narukami Shrine. When held, it allows the bearer to briefly sense the lingering 
                aspirations of a confiscated Vision, offering guidance or comfort. However, prolonged use risks trapping the user in 
                echoes of another's unfulfilled dreams, blurring the line between their own will and borrowed longing.
            """
        }
    ]
}

TEST_DATA_TEXT1_SHORT = {
    "path_to_graph": "assets/graphs/elden_ring_lore.json",
    "path_to_text": "assets/documents/ELDEN RING LORE.txt",
    "text_summary": """
        The Greater Will is an Outer God whose power manifests through the Elden Ring and the Erdtree. 
        Queen Marika becomes its vessel and establishes the Golden Order to enforce its will. 
        Before this, dragons ruled the Lands Between but declined after their god abandoned them. 
        With the rise of the Golden Order, beings tied to the primordial “crucible” (like Omens) became persecuted.
        Marika wages wars, including against the Fire Giants, and allies with figures like Hoarah Loux (Godfrey). 
        She and her other self, Radagon, are two aspects of one being. 
        Their children, including Malenia and Miquella, are cursed. Other Outer Gods also influence events.
        Ranni rejects the Greater Will and orchestrates the Night of the Black Knives, killing Godwyn. 
        In response, Marika shatters the Elden Ring, triggering the Shattering war among demigods, each seeking power or change.
        As the Greater Will’s influence weakens, the Tarnished return to the Lands Between, 
        aiming to restore or redefine the world’s order.
    """,
    "language": "en",
    "tasks": [
        {
            "category": "quest",
            "query": "Create a new Elden Ring-style side quest involving a Tarnished helping a minor faction or individual affected by the Shattering. The quest should involve moral ambiguity, hidden truth, and a choice between loyalty to the Golden Order or defiance of it. Do not reuse known canonical events.",
            "reference": """
                A Tarnished encounters a wandering scholar from a minor ruined order who claims to have discovered fragments 
                of forbidden knowledge about the Shattering. The scholar asks for protection while traveling to an ancient site where truth 
                about the Erdtree’s origins may be revealed. Along the journey, it becomes unclear whether the scholar seeks 
                enlightenment or intends to destabilize remaining order. At the destination, the Tarnished must choose to either help seal 
                the knowledge away or release it to the world, risking further chaos.
            """
        }
    ]
}

TEST_DATA_TEXT2_SHORT = {
    "path_to_text": "assets/documents/Leisure Suit Larry 6.txt",
    "path_to_graph": "assets/graphs/leisure_suit_larry.json",
    "text_summary": """
        Larry Laffer arrives at a luxury spa after losing a dating show, where he is treated poorly as a non-paying guest. 
        Throughout his stay, he helps several women, including Gammie with her weight-loss treatment, Shablee by finding her a dress for a date, 
        and Char with gathering items—though many encounters end in rejection or embarrassment. He also meets Cavaricchi and Burgundy, who invite 
        him to a sauna but exclude him, and Thunderbird, a dominatrix who humiliates him. Larry assists Merrily with her dream of bungee jumping, only 
        to accidentally fall himself. Despite repeated mishaps, Larry persists. In the end, after completing tasks for Shamara, he finally gains her affection, 
        concluding his misadventure on a successful note.
    """,
    "language": "en",
    "tasks": [
        {
            "category": "quest",
            "query": "Design a new comedic adventure puzzle sequence set in the same luxury spa. The quest should involve helping a different eccentric guest achieve a risky personal goal by stealthily acquiring and duplicating a restricted access item. The sequence should culminate in an ironic, highly public mishap for the protagonist while the guest finally attains their desire, maintaining the game's signature blend of slapstick and puzzle-solving.",
            "reference": """
                Larry assists Merrily Lowe at the pool bar by obtaining the diving tower key from the lifeguard, 
                making an impression of it in a bar of soap, filing a copy with a bastard file, and returning it to her. 
                They climb to the bungee platform, where she shares her "Words of Wisdom," but Larry trips over the cords and falls, 
                accidentally broadcasting his naked descent to the entire spa.
            """
        }
    ]
}


TEST_DATA_TEXT3_SHORT = {
    "path_to_graph": "assets/graphs/inazuma_lore.json",
    "path_to_text": "assets/documents/INAZUMAS MAIN QUEST.txt",
    "text_summary": """
        The Traveler arrives in Inazuma, a nation ruled by the Electro Archon, the Raiden Shogun, 
        who enforces isolation and the Vision Hunt Decree—confiscating Visions from citizens, stripping many of their ambition and purpose. 
        This causes widespread unrest and resistance.

        The Traveler meets allies like Ayaka and Thoma and witnesses the growing oppression. A resistance led by Sangonomiya Kokomi 
        fights the Shogunate, while the Fatui secretly manipulate events to destabilize Inazuma further.

        It is revealed that the true Archon, Raiden Ei, has retreated into the Plane of Euthymia, leaving a puppet Shogun to enforce her ideal 
        of "eternity"—a world without change or loss.

        The Traveler confronts the Shogun and then Ei herself within her inner realm. Through a duel and witnessing 
        the will of Inazuma's people embodied in their Visions, Ei realizes the harm of her ideology.

        She abandons the Vision Hunt Decree, restores the Visions, ends the civil conflict, and begins guiding Inazuma toward a new future.
    """,
    "language": "en",
    "tasks": [
        {
            "category": "quest",
            "query": "Design a new side quest set in Inazuma where the Traveler helps a former Vision bearer cope with the loss of their aspirations after the Vision Hunt Decree. The sequence should involve investigating rumors of a hidden sanctuary where confiscated Visions are kept, navigating moral choices about whether to expose the truth or protect those who still hope, and culminating in a quiet moment of reflection rather than combat.",
            "reference": """
                In the quest "The Meaning of Meaningless Waiting," the Traveler meets Tejima, a former samurai whose Vision was confiscated. 
                Without his Vision, he struggles to remember why he stayed in Konda Village. The Traveler helps him find a letter revealing 
                he waited for a lover who never returned. Tejima chooses to remain, finding new meaning in patience rather than action.
            """
        }
    ]
}

TEST_DATA_LORE_DESCRIPTION = {
    "path_to_graph": "assets/graphs/generation_tests/elden_ring_lore_graph.json",
    "path_to_text": "assets/documents/generation_tests/elden_ring_lore.txt",
    "text_summary": """
        The Lands Between were originally ruled by dragons and a primordial Erdtree known as the Crucible, but the Outer God known as the Greater Will sent the Elden Beast to establish its influence. Queen Marika became its vassal, bearing the Elden Ring, and founded the Golden Order. Over time, Marika waged wars against the Fire Giants, the Carian royals, and others, while the Greater Will’s emissaries (the Two Fingers) enforced its desires. Marika’s first consort, Godfrey (Hoarah Loux), was eventually exiled and became the first Tarnished. Her second consort, Radagon, is later revealed to be Marika’s other half. Their children—Malenia, Miquella, and others—suffered curses, and Miquella ultimately rejected the Golden Order, creating the Haligtree as a refuge for the oppressed.
        Ranni, daughter of Radagon and Rennala, orchestrated the Night of the Black Knives, stealing the Rune of Death to kill the demigod Godwyn. Marika, possibly complicit or grief-stricken, shattered the Elden Ring, leading to her imprisonment within the Erdtree. The Shattering War followed, with demigods like Radahn, Malenia, Morgott, and Mohg fighting for power. Malenia’s battle with Radahn devastated Caelid with Scarlet Rot, while Mohg kidnapped Miquelia. Meanwhile, Ranni destroyed her own body to avoid becoming the Greater Will’s vassal, and other Tarnished—including Fia, Goldmask, the Dung Eater, and Sir Gideon—returned to the Lands Between, each seeking to reshape the Elden Ring according to their own vision of order, chaos, or death.
    """,
    "language": "en",
    "tasks": [
        {
            "category": "character description",
            "query": "Write a short, unstructured character description of a completely new character whose path crosses the player's (Tarnished) path several times. Everything for this character happens after all of the events that took place in this world. The character's goal is to become a great champion, a strong warrior. He should look as unusual as possible, but this unusualness should be maximally simple (for example, take inspiration from the idea of taking some inanimate object — one, simple object — and basing a character on it). At the same time, the character should evoke sympathy and mild condescension, as well as admiration for his perseverance. Write a descriptive text, mentioning the character's personality in literally one sentence, his appearance in one sentence, and one sentence describing his backstory. Then write a short paragraph (3-5 short sentences) that reveals how the player encounters this character, under what circumstances, how they interact, and what the character does during these encounters.",
            "reference": """
                Alexander, Warrior Jar
                Alexander is a living jar who travels the Lands Between, honing his strength to become a great warrior. Alexander was created to be a warrior vessel, and the remains of many great warriors reside within him. He originally came from Jarburg, a hidden village of living jars in Liurnia of the Lakes; however, he vowed never to return, claiming that "the path of champions must be trod alone". Despite his occasional incompetence and need for rescue, Alexander is unfailingly optimistic, determined, and honorable—viewing every setback as a step toward becoming a "grand champion". He looks like a pot filled with some sort of meat, with long black arms and short legs.
                On his journey to become a grand champion, he seeks out Redmane Castle in Caelid, following rumours of a festival of combat. On his way to the festival, he crosses paths with the Tarnished multiple times, often requiring their assistance to get out of unfortunate situations such as being stuck in potholes. He participates in the Radahn Festival, a brawl between several champions and Starscourge Radahn. Alexander survives the ordeal, however, he suffers damage to his body and is forced to hide for the remainder of the battle. Undeterred, he replenishes himself with the bodies of mighty warriors littering the battlefield, vowing to grow even stronger. Next, Alexander travels to Mt. Gelmir, hoping to temper his body in the sea of fire, such that it would never crack again. Finding the lava to be lukewarm and insufficient to strengthen his vessel, he sets his sights upon the Mountaintops of the Giants, seeking out the flame of ruin. Alexander travels to the mountaintops, where he witnesses the Tarnished's fight with the Fire Giant. He then finds himself in Farum Azula, where he encounters the Tarnished for a final time. Impressed by his companion's feats of strength, he requests a duel. The Tarnished emerges victorious and Alexander accepts his defeat, asking the Tarnished to take his innards with his final words.
            """
        },
        {
            "category": "character descripion",
            "query": "Write a structured character description for a game character who, over the course of the game, learns a skill and becomes a hub for the player to upgrade or trade something. This character should be encountered by the player at the very beginning of the game, after all the possible events have already taken place, and the character's development should be visible after the player finds them. Write 4 short paragraphs (3–7 short sentences each, DO NOT GO BEYOND THESE LIMITS) that follow this structure: Character's Story (a coherent text about where the character comes from, what they did, who their companions were), Character's Appearance (what the character looks like, what they wear), Character's Personality (how the character behaves, how they feel about themselves, what they think about), Character's Motivation (what drives the character at different points in time). Format the text as a HEADER IN CAPS followed by the paragraph description.",
            "reference": """
                Roderika, Spirit Tuner
                CHARACTER'S STORY
                Roderika is a Tarnished noblewoman who, despite having never seen the guidance of grace, travelled to the Lands Between along with her companions. After arriving, those companions fell victim to Godrick, becoming subjects for his grafting. Roderika alone survived, taking refuge in Stormhill Shack. It is here that she meets the Tarnished. She beseeches the Tarnished to bring a message to her grafted companions, whom she refers to as chrysalids, assuring them that she loves them and will join them soon.
                Roderika then heads to Roundtable Hold. Here, she becomes acquainted with Smithing Master Hewg, who takes notice of her gift for spirit tuning and is reminded of a spirit tuner he knew long ago, whose eyes were of the same hue as Roderika's. Reluctantly, Hewg agrees to teach Roderika everything he knows about spirit tuning.
                Roderika remains at Roundtable Hold, where she hones her newfound talent and eventually becomes a fully-fledged spirit tuner. Her ability to sense spirits allows her to notice the Dung Eater's presence, which she warns the Tarnished about.
                CHARACTER'S APPEARANCE
                Roderika is a young woman with short, tousled ash-blonde hair, pale skin, and large eyes of a pale light blue color. She wears a large Crimson Hood that covers much of her head and shoulders. The hood is a deep, rich red, which stands out vividly against the drab, muted colors of Limgrave. Beneath this, she wears a simple white noblewoman cloth, containing a light tunic and trousers. 
                CHARACTER'S PERSONALITY
                Roderika is defined by a profound sense of cowardice and self-doubt, which she openly acknowledges. Initially, she is paralyzed by fear, describing herself as a "craven" and a "milksop" who is too terrified to face the grafting she witnessed. Despite this, she possesses a deep well of compassion and loyalty, demonstrated by her love for her fallen companions and her desire to send them a message of comfort. As she finds her purpose in spirit tuning, her personality blossoms; she becomes more grounded, perceptive (able to sense the Dung Eater's curse), and quietly determined. Her compassion extends to Hewg, whose gentle nature she alone recognizes, and her loyalty solidifies into unwavering resolve as she refuses to abandon him in the burning Roundtable Hold. She remains humble, preferring to support others from the sidelines rather than seek glory for herself.
                CHARACTER'S MOTIVATION
                Roderika's primary motivation evolves from mere survival to finding a meaningful purpose to honor the sacrifice of her companions. Initially, she seeks only to join her "chrysalids" (her grafted friends), believing herself useless. However, discovering her gift for spirit tuning gives her a new reason to live: to ease the suffering of spirits, including her fallen men, by honing her craft. Her ultimate motivation becomes intertwined with loyalty and redemption. She is driven to free Hewg from what she perceives as Queen Marika's "fearsome curse," even trying to persuade him to leave the hold. When he refuses, she is motivated to stay with him out of gratitude, returning the kindness he showed her. Finally, she channels her grief and determination into a single, desperate plea: for the Tarnished to become Elden Lord and use Hewg's god-slaying weapon to kill Marika, the source of the curse that binds her mentor and caused so much suffering.
            """
        },
        {
            "category": "character description",
            "query": "Write a short (no more than 15 short sentences) unstructured character description of a character who has faithfully served Ranni the Witch for many years, from the moment she was chosen as an Empyrean. The character is loyal to Ranni in any situation and serves as a knight-protector. Try to give them an unusual appearance that reflects their eternal devotion to their mistress (you can make them half-animal, for example, but keep humanoid features). Write one sentence each for the character's appearance, personality, and describe their history and history of their service to Ranni. Do not structure the text; make it a single descriptive paragraph without headers. Just a short description.",
            "reference": """
                Blaidd the Half-Wolf
                A guarded and boorish yet fiercely loyal half-wolf, Blaidd was a shadowbound beast, created by the Two Fingers to serve Lunar Princess Ranni when she was chosen as an Empyrean. He is a wolf-human hybrid with natural wolf-like facial features, pointed ears, and canine teeth. He wears a distinctive blue-tinged rugged chest piece with black greaves and gauntlets, a heavy fur cape, and wields the Royal Greatsword.
                The wolf was the beast of the Carian royal covenant and irrefutably symbolized the moon's pride, and so he was approved and raised by Rennala, Queen of the Full Moon as Ranni's stepbrother.
                Blaidd was Ranni's unwavering blade. Like all shadowbound beasts, he could not truly die and was incapable of treachery against his master, compelled to enact her will. Unbeknownst to him, however, an accursed failsafe within shadows overrode this directive should their master turn against the Two Fingers; should Ranni rebel against her prescribed fate, Blaidd’s unwavering loyalty would twist into madness, becoming a Baleful Shadow—a dire threat to Ranni herself.
                When Blaidd swore an oath to serve no master other than Ranni the Witch and her Dark Moon, a cold frost magic graced his Royal Greatsword.The cold bothered him, nonetheless.
            """
        },
        {
            "category": "character description",
            "query": "Write a short (no more than 15 short sentences) unstructured NPC character description: something sad, about family relationships and two family members caring for each other, where the character you are describing dies at the end. All these events happens after the whole world story and all events of this story. Write one sentence each for the character's appearance, personality, and describe their history: how they lived, what condition they are found in by the player, what happens afterward. Do not structure the text; make it a single descriptive paragraph without headers. Just a short description. It is a small side activity, little sad story, so it is not tied to main characters, however, it happens in a game world of Lands Between and it is connected with player's Tarnished, as it is part of his path.",
            "reference": """
                Irina of Morne
                Irina is the daughter of Edgar, castellan of Castle Morne. Her eyesight has been weak since birth, and she wears a blindfold around her eyes. Irina is a gentle, fearful soul whose love for her father outweighs her terror, a young woman defined by quiet courage in the face of despair. She appears as a fragile figure in a bloodstained traveling maiden's robe.
                Irina lived in Castle Morne with her father, until the Misbegotten servants rebelled and overran the castle. Edgar secreted Irina out of the castle, however the Misbegotten pursued her and killed her companions.
                Irina encounters a Tarnished at the Bridge of Sacrifice on the Weeping Peninsula, and petitions them to rescue her father from Castle Morne, handing them a letter to deliver to him.
                The Tarnished travels to Castle Morne and passes the letter onto Edgar. Once the Leonine Misbegotten has been defeated, Edgar leaves the castle to reunite with his daughter, vowing to devote his remaining days to her. However, by the time he reaches her location, Irina is already dead. Edgar swears vengeance on those responsible for her death, setting him on a path that eventually leads to his corruption by the Flame of Frenzy.
            """
        },
        {
            "category": "character description",
            "query": "Write a short description for an NPC Finger Reader who is both a merchant for a special currency and an interpreter of the Fingers' will. She stands near the Two Fingers and deciphers their messages for the Tarnished (the player). Write one sentence for her appearance (short), one for her personality, and also describe how the interaction with her changes as the player (Tarnished) kills the demigods who fought for the right to become Elden Lord, one by one. All this happens after all of the world's story events. Do not structure the text; make it a single descriptive paragraph without headers, and keep it short, no more than 10 sentences.",
            "reference": """
                Finger Reader Enia
                Enia is a long-lived Finger Reader, much like the Finger Reader Crones scattered across the Lands Between. She interprets the words of the Two Fingers, envoys of the Greater Will. Enia remains in Roundtable Hold alongside the Two Fingers, offering words of wisdom throughout the Tarnisheds' journeys. She is a wizened crone who wears heavy robes befitting her office as a Finger Reader and carries a large club-like staff. Enia begins as a dutiful interpreter who faithfully relays the Fingers' words in formal, ceremonial language, but as events unfold and the Fingers fall silent, she shifts toward pragmatic acceptance and dry humor, ultimately encouraging the Tarnished to forge their own path.
                The Tarnished meets Enia in the Two Fingers' audience chamber at Roundtable Hold after claiming a Great Rune, where she imparts the Fingers' wisdom—that they must gather more Great Runes to repair the Elden Ring—and offers to unlock the power within Remembrances gained from defeating powerful foes. After the Tarnished defeats Morgott and the Erdtree remains sealed, the Two Fingers fall dormant while consulting the Greater Will; Enia warns against burning the Erdtree and unleashing the Rune of Death, but ultimately advises the Tarnished to do what you believe is right before she too goes dormant.
            """
        },
        {
            "category": "location description",
            "query": "Write a short description for a game hub where the player (the Tarnished) can find everything they need (merchants, a blacksmith) to exchange collected currency and upgrade. This should be a cozy place, filled with many NPCs, more like a small castle. Here, all Tarnished can find help and rest during their journey, and the Two Fingers are also located here. Describe what the location looks like, what it is, what rooms it has, where it is located, how it was created, and why. In your answer, write a short (no more than 15 sentences) unstructured text with the description.",
            "reference": """
                The Roundtable Hold
                The bustling hub area that exists outside of the Lands Between. A place to mingle with other adventurerers, craftsmen, resupply and discover a trove of blessings and rooms of secrets.
                A pocket realm tied to the Erdtree's Grace, the Roundtable Hold exists to aid the Tarnished on their quest for the Elden Ring, with Smithing Master Hewg as its foundation. It is a copy of Leyndell's Fortified Manor, once a gathering place for great champions seeking the Two Fingers' divine wisdom. A bustling community of mighty Tarnished likewise convened in the Hold in the distant past; however, it is now a shadow of its former glory, with its present occupants lacking the venerable status and lofty aspirations of those past heroes.
                The Roundtable Hold is a multi-level hub with a main floor and cellar, centered around the Lobby containing the Table of Lost Grace site of grace where key NPCs like Gideon, Corhyn, and eventually Roderika gather. The Northern Wing holds Gideon's study and the Twin Maiden Husks (a bell-bearing shop), while the Eastern Wing leads to Fia's room, Hewg the blacksmith, and the armory which unlocks after a quest. Other areas include a Balcony, an invasion-triggering Hall, living quarters with the Cipher Pata, a pantry where Nepheli moves post-quest, locked rooms requiring Stonesword Keys, and the Audience Chamber with the Two Fingers and Enia, which opens after obtaining a Great Rune.
            """
        },
        {
            "category": "location description",
            "query": "Write a description for the location of Raya Lucaria Academy. Describe the building where this faction is located: give a description of its location and a general description of what happens inside (Academy — meaning, most likely, education, so focus on that). Then follow this structure: write an APPEARANCE block, describing in it what the location looks like, its atmosphere, and its contents (3–4 sentences). Next, write a HISTORY block, telling the history of how this location came to be, how it is connected to Rennala and her story, and how it experienced the events of Rennala's history. (Describe the history in detail, in 2–3 paragraphs of 3–4 sentences each.)",
            "reference": """
                The Academy of Raya Lucaria
                The Academy of Raya Lucaria sits on a plateau in the middle of the great lake of Liurnia, and is the place of study for glintstone sorcerers of the Lands Between. This area is accessed by using an Academy Glintstone Key to pass through either of the two gates marked with the blue seal.
                The Academy offers everything from basic training in sorcerery to advanced, specialized study. Higher learning is accessible to only noteworthy scholars, who are then permitted to don stone Glintstone Crowns, modeled after notable sorcerers in their chosen conspectus.Azur and Lusat, regarded as founding glintstone sorcerers, founded the Karolos and Olivinus Conspectuses, respectively.The majority of Glintstone Crown sorcerers found within the Academy pursue one of these two founding conspectuses.
                APPEARANCE
                Raya Lucaria Academy is a massive, gothic-style castle sealed behind a magical barrier at the South Gate, requiring a Glintstone Key to enter. Inside, high towers and crumbling rooftops stretch over a central plaza, a chapel filled with enemies, a vast graveyard, and narrow, library-like corridors packed with glintstone sorcerers. The atmosphere is one of eerie decay and magical claustrophobia, with blue crystal formations, flying books, and haunted silence broken only by masked scholars and Iron Virgin dolls. Key areas include the Church of the Cuckoo, the Schoolhouse Classroom, and the Grand Library—home of Rennala, Queen of the Full Moon. Unlike Stormveil's chaotic fortifications, Raya Lucaria feels trapped in time, beautiful yet absurd, with fog and rain over Liurnia adding to its isolated, end-of-the-earth melancholy.
                HISTORY
                The founding of the Academy of Raya Lucaria was credited to the primeval sorcerers Azur and Lusat. Their study of the Primeval Current shook the academy when they began to transfigure fledgling sorcerers into Graven Masses—the seeds of stars.
                Long ago, Rennala charmed the Academy with her lunar magic. She became the Academy's master and established the house of Caria as royalty. When Rennala assumed governorship over the Academy, she heavily limited the study of the primeval current due the dangers it presented and subsequently exiled Azur and Lusat. This change outraged Sorceress Sellen, who began to transform her peers into Graven Schools. Though the Academy expelled Sellen and denounced her methods, they shared her core sentiment, and their thirst for knowledge would eventually lead to the downfall of the Carians influence over the Academy.
                The combined forces of the Knights of the Cuckoo: the Cuckoo Knights, the Raya Lucaria Soldiers and their Foot Soldiers were hired by the Academy at some point. As payment for their service, the Academy taught them martial Glintstone techniques like Scholar's Armament and Scholar's Shield. The design of the Cuckoo bird upon their armor is mused to represent the Cuckoo's refusal to be mere servants of the Academy. The Cuckoo were given free reign to wage war as they pleased by the Academy, and became known for their rapacious ways.
                When Rennala's husband Radagon left her to become the second Elden Lord, Rennala's heart went with him. With Rennala a grieving shell of her former self, the Academy then realised that she was no champion after all. She still clutches the amber egg given to her by Radagon, which contains the Great Rune of the Unborn. No longer in any state to defend herself, the Academy took the opportunity to lock Rennala in their Grand Library and staged a rebellion against the Carians, so that they could pursue unlimited study once more. The resulting conflict between the Acadamy's forces, and those of the Carians, waged across Liurnia, becoming an effective Liurnian civil war. Ultimately, the Acadamy would achieve their goals. Without their queen, and her successors absent, the Carians fell into decline, and their influence in the Academy's policies dissipated.
                After the Shattering of the Elden Ring, the Academy declared neutrality in the ensuing war and cast repelling seals upon its entrances, blocking outside access to anyone not in possession of a Academy Glintstone Key. Such a key may be found on an island west of the Academy, guarded by Glintstone Dragon Smarag.
            """
        },
        {
            "category": "location description",
            "query": "Write a location description for a manor that was the home of Praetor Rykard. Describe the manor's appearance, atmosphere and its history: how the praetor used it, how the manor is connected to his history and the events of his life, what the manor looks like after all these events, and who remains in it after everything that happened. Write a total of 4 paragraphs of 3–5 sentences each, and the text should be unstructured: plain text without headings.",
            "reference": """
                Volcano Manor
                Volcano Manor is a vast, ominous manor with grand halls and a majestic throne room. It is located at the summit of the crater of Mount Gelmir, surrounded by sheer cliffs. Long ago, the manor was overseen by Praetor Rykard and his wife Tanith. The Praetor served as the head of the inquisitors, and so he turned his own home into both a torture chamber and a prison.
                In the dungeons of the manor, within the crater of the volcano, the inquisitors tortured nobles using a variety of methods. For example, using a special candlestand, the victim's body was first pierced with numerous spikes, and then the wounds were cauterized with fire. The smell of burnt blood drove the person to despair. That weapon was the product of a highly sophisticated mind. Inquisitor Ghiza, for his part, used a giant mechanical wheel for torture, modeled after the weapons of the Iron Virgins. And to inflict even greater suffering on the victims, masks were placed on their heads, which exacerbated the person's fear and pain. Once the Black Dumpling came into play, the torturer no longer needed answers—only suffering. Do not hope for mercy.
                During the time of the Shattering, Praetor Rykard rebelled and embraced blasphemy, and everyone in Volcano Manor supported the newly proclaimed Blasphemous Lord. Soon, however, the army of Leyndell laid siege to the manor. The bloodiest battle of the Shattering era ended in defeat for the capital; no one managed to capture Volcano Manor.
                Soon, Rykard offered himself to the great serpent to be devoured, but by doing so he lost his knights, who decided to kill him. However, they were unsuccessful, and the only one who remained by Rykard's side afterwards was his wife Tanith. She began to rule Volcano Manor alone and created an organization of rebels who opposed the Two Fingers and the Erdtree. Their home became Volcano Manor, where snake-men also lived in secret.
            """
        },
        {
            "category": "location description",
            "query": "Write a short description for a region, that located beneath the Lands Between. This is a river where one can find Nokron and Mohgwyn Dynasty Mausoleum. This should be beauriful underground night location, connected with old dynasties existed before the Erdtree. Write a description for this location: what is located there, describe the atmosphere, appearance, history and connections of this location with world story, groups and characters. Write 4 paragraphs from 3 to 4 sentences.",
            "reference": """
                Siofra River
                One of the two great rivers that flow beneath the Lands Between. Siofra is said to be the grave of civilizations that flourished before the Erdtree, and evidence of this can be found in the many ancient structures that litter the vast, subterranean space. The towering remains of an ancient dynasty are prominent throughout the area, inhabited by the Claymen who once served as the dynasty's priests.
                The lower portion of Siofra River can be accessed via the Siofra River Well in Limgrave's Mistwood. This wooded area is home to spectral Ancestral Followers, who live alongside a diverse assortment of wildlife. The remains of a great, horned beast can be found in the Hallowhorn Grounds. Interacting with the remains after lighting the flames scattered throughout the area will facilitate an encounter with an Ancestor Spirit. This area also connects to Caelid via the Deep Siofra Well.
                Towering over the woodland is Nokron, one of the Eternal Cities inhabited by the Nox. Nokron is initially inaccessible, however a route is opened when a star falls upon Limgrave, following the defeat of Starscourge Radahn. The Fingerslayer Blade, a treasure of Nokron said to be able to harm the Greater Will and its vassal Fingers, is found within Night's Sacred Ground, beneath a chair-crypt that holds a vast corpse. This upper region also houses another Hallowhorn Grounds, from which the Regal Ancestor Spirit can be encountered. Nearby, the aqueduct hides a coffin which can be used to access Deeproot Depths, guarded by a pair of Valiant Gargoyles.
                To the east, on a separate island within the vast cavern, are the ruins of an ancient palace. It is here that Mohg, Lord of Blood has established the seat of his coming Mohgwyn Dynasty, and all the nightmares it may bring.
            """
        },
        {
            "category": "location description",
            "query": "Create a very short (5-7 sentences) description of a location, which is after all events of a world is a residence of Godrick the Grafted. Describe history of this place, its current state and its appearance in an unstructured text without headers.",
            "reference": """
                Stormveil Castle
                Stormveil Castle perches atop a cliff overlooking Limgrave. Currently, it is the stronghold of the Demigod Godrick the Grafted, but long ago it was ruled over by an old king, back when the true storm raged. The castle is heavily guarded and patrolled by Banished Knights, Exile Soldiers, and warhawks with blades grafted to their talons. The exterior of the castle is mottled with craters lined with thorns, which have also spread to the soldiers within. It is said that the source may be hidden deep within the castle.Stormveil Castle is a sprawling, Gothic-style legacy dungeon perched on the cliffs of Stormhill in Limgrave, acting as the primary stronghold of Godrick the Grafted. It is characterized by a gloomy atmosphere, with parts of it covered in sickly, organic thorns reflecting its corruption
            """
        },
        {
            "category": "item description",
            "query": "Create a very short (3 sentences) and very simple description for an item: it is an unfinished Miquella's needle, one of several, that was being created as a powerful ritual item, but remains with a hald of its power.",
            "reference": """
                One of the unalloyed gold needles that Miquella crafted to ward away the meddling of outer gods.
                Capable of subduing the flame of frenzy if inherited, allowing one to cheat fate and avoid becoming Lord of Frenzied Flame.
                However, the needle is as yet unfinished and can only be used in the heart of the storm beyond time said to be found in Farum Azula.
            """
        },
        {
            "category": "item description",
            "query": "Create a very short 2-sentence description for a doll that ressembles Ranni the Witch. It is a very precise doll, write about it.",
            "reference": """
                A doll resembling Ranni the Witch. From head to toe, every detail is perfect.
                This unresponsive doll seems pleasantly cool.
            """
        },
        {
            "category": "item description",
            "query": "Create a short description (4 sentences) for a Cursemark that had to be circle and created when first demogod dies, but two demigods - Ranni and Prince of Death - dead at the same moment, so this cursemark was broken into two half-wheels.",
            "reference": """
                Cursemark carved into the discarded flesh of Ranni the Witch. Also known as the half-wheel wound of the centipede.
                This cursemark was carved at the moment of Death of the first demigod, and should have taken the shape of a circle.
                However, two demigods perished at the same time, breaking the cursemark into two half-wheels.
                Ranni was the first of the demigods whose flesh perished, while the Prince of Death perished in soul alone.
            """
        },
        {
            "category": "item description",
            "query": "Create a description for an item that is a pass to Academia Raya Lucaria: describe its appearance and its very short description for a game. Write a text with two headers: APPEARANCE and ITEM DESCRIPION, both blocks are less than 4 sentences.",
            "reference": """
                The Academy Glintstone Key
                APPEARANCE
                A large, ornate, glowing blue crystalline key held by a dead sorcerer behind the Glintstone Dragon Smarag.
                ITEM DESCRIPTION
                Key to the seals binding both gates to the Academy of Raya Lucaria.
                Activates warp magic bound within the seals.
                A glintstone key will remember its user, meaning once used it can never be passed on to another. The academy does not welcome the indolent.
            """
        },
        {
            "category": "item description",
            "query": "Create a description for an item that Lunar Princess Ranni had to give to her destined consort. Describe its appearance and its very short description for a game. Write a text with two headers: APPEARANCE and ITEM DESCRIPION (two words about appearance, meaning of an item, some additional information), both blocks are less than 5 sentences.",
            "reference": """
                Dark Moon Ring
                APPEARANCE
                A silver-toned band featuring a dark, leaden full moon crest. It symbolizes the cold oath of Lunar Princess Ranni and is described as having a cloudy or dark, occult, and icy appearance.
                ITEM DESCRIPTION
                Ring depicting a leaden full moon. Symbolic of a cold oath, the ring is supposed to be given by Lunar Princess Ranni to her consort.
                Ranni is an Empyrean, meaning her consort would by rights earn the title of lord.
                A warning is engraved within; "Whoever thou mayest be, take not the ring from this place, the solitude beyond the night is better mine alone.
            """
        },
        {
            "category": "quest",
            "query": "Create a quest for a side character who can be met in a manor - residence of Rykard. This is a quest about Rya, whose full name is Zoraya, and who lives in a Volcano Manor with Tanith, the wife of Rykard. Rya can transforms into a snake, that must be revealed during the quest. Quest must contain six stages: initial enconter and task, reaching new location of Rya and meeting her again, then some progression steps, and finally coming to final location and choosing some possibilites in conclusion. Structure quest description as some steps using a list: describe player's actions (what they should do) and character's answer actions. For every stage of quest create minimum three steps (can be simple like finding person-speaking to him-receiving smth). Structure as lists of actions with headers, describing stages.",
            "reference": """
                Rya/Zorayas Questline — Descriptive Overview
                Initial Encounter & Necklace Recovery
                    1. The Tarnished first encounters Rya at Liurnia of the Lakes, seated under a pavilion next to the Birdseye Telescope north of Laskyar Ruins.
                    2. After exhausting her dialogue, she requests assistance in retrieving a stolen necklace.
                    3. The thief, Blackguard Big Boggart, can be found at the Boilprawn Shack to the northwest. He offers to sell the necklace for 1,000 Runes; alternatively, it can be obtained by defeating him (though this action locks the Tarnished out of Boggart's own questline).
                    4. Upon returning the necklace, Rya rewards the Tarnished with a Volcano Manor Invitation and provides information about the two main routes to Altus Plateau.

                Arrival at Altus Plateau & Volcano Manor
                    1. Once the Tarnished reaches Altus Plateau, Rya can teleport them directly inside Volcano Manor. Her location on the plateau depends on the route taken:
                        - If the Tarnished ascended via Ruin-Strewn Precipice first, she appears at the lower part of Lux Ruins (up the stairs from the Erdtree-Gazing Hill Site of Grace).
                        - If the Grand Lift of Dectus was used first, she appears at the top of the lift, to the left of the stairs.
                        - Note: If Rya does not appear at either location, re-triggering the Grand Lift ascent may spawn her. If Leyndell was reached via Fia's questline first, she will not appear at these locations at all.*

                Volcano Manor Progression — Snake Form
                    1. After joining Volcano Manor and completing the first assassination contract for Lady Tanith (Old Knight Istvan), Rya relocates to a different room within the Manor and appears in her serpent form.
                    2. Speaking to her in this form advances the questline. Note: If all Volcano Manor contracts are completed to the point of being transported to Rykard, Lord of Blasphemy via Tanith, Rya's dialogue related to her optional objectives becomes unavailable.*

                Secret Passage & Prison Town Church
                    1. After completing the second Tanith contract (assassination of Rileigh the Idle), Rya returns to human form in the same room. She mentions hearing noises from an adjacent room and asks the Tarnished to investigate.
                    2. The Tarnished discovers an illusory wall in the room to her right; passing through it leads to the Prison Town Church Site of Grace.
                    3. Returning to Zorayas (Rya's true name) and reporting the discovery progresses the quest. Note: The doors exiting Prison Town Church must be opened to continue.*

                Optional Branch — Serpent's Amnion
                    1. (Optional) Speaking to Lady Tanith and selecting the "Zorayas' troubles" dialogue option unlocks an additional objective.
                    2. Defeating the Godskin Noble in the Temple of Eiglay yields the Serpent's Amnion.
                    3. Giving the Serpent's Amnion to Zorayas causes her to disappear from the Manor after the Tarnished rests at a Site of Grace.
                    4. (Optional but crucial for one ending path) Speaking to Lady Tanith again after Zorayas' disappearance adds the dialogue option "Zorayas' absence". Tanith provides a Tonic of Forgetfulness intended for Zorayas. Note: If all assassination contracts were completed prior, the Tonic cannot be obtained until after Rykard is defeated and Tanith is spoken to again. If Zorayas has already left the Manor, the Tonic will be found on her chair.*

                Final Location — Legacy Dungeon Alcove
                    1. Zorayas relocates to a small alcove within the Legacy Dungeon interior, adjacent to a lava pit filled with cages and skulls, just before a rope ladder.
                    2. Access routes:
                        - Via the wooden elevator next to the Temple of Eiglay: disembark at a hidden doorway just below the elevator's highest point, proceed through the room, exit a window, and jump across a lava floe to the room directly opposite.
                        - If the elevator is inactive: jump from the balcony on the top floor of the Temple of Eiglay to reach the lava floe area (this also unlocks the elevator for future use).
                        - Alternative path: Take the lift within the Temple of Eiglay (southeast of the grace) upward, exit through the door to a balcony, descend to the lava pit with slugs, cross the rocky bridge, climb the path, hop across the lava to a window on the left (beware the Iron Virgin), proceed past the man-serpent on the bridge, descend the stairs, and locate a ladder in a left-side window leading to the alcove.

                Conclusion — Three Possible Outcomes
                At her final location, the Tarnished may choose one of three actions:
                1.  Defeat Zorayas: She transforms back into serpent form and drops Daedicar's Woe.
                2.  Leave her alive and return after defeating Rykard: She offers new dialogue; upon resetting the area, she is gone, leaving behind Daedicar's Woe and Zorayas' Letter.
                3.  Administer the Tonic of Forgetfulness: She falls asleep. After Rykard's defeat and the departure of the Manor's inhabitants, she returns to her original location within the Manor. Exhausting her dialogue and reloading the area yields Daedicar's Woe in her place.
            """
        },
        {
            "category": "quest",
            "query": "Create a quest for a side character who can be met exhausted and scared in Stormhill Shack, near the residence of Godrick. This is a quest about Roderika, who is a noble Tarnished and lost her companions bacause of Godrick's grafting. During this quest, she find herself and becomes Spirit Tuner. Quest must contain three stages: initial enconter and task, reaching new location of Roderika and meeting her again, then help in her development, and finally coming to final location and choosing some possibilites in conclusion. Structure quest description as some steps using a list: describe player's actions (what they should do) and character's answer actions. For every stage of quest create minimum three steps (can be simple like finding person-speaking to him-receiving smth). Structure as lists of actions with headers, describing stages.",
            "reference": """
                Roderika Questline — Descriptive Overview
                Initial Encounter at Stormhill Shack
                    1.The Tarnished first encounters Roderika at Stormhill Shack, where she appears as a timid noblewoman grieving for her fallen companions.
                    2. Exhausting her dialogue multiple times rewards the Tarnished with the Sitting Sideways pose, followed by the Spirit Jellyfish Ashes.
                    3. Roderika requests that the Tarnished deliver a message to her "chrysalids" (grafted companions) and retrieve a memento from them.
                    4. The Chrysalids' Memento can be looted from a pile of corpses within Rampart Tower in Stormveil Castle.
                    5. Returning the memento to Roderika while she remains at the shack progresses her storyline.
                Transition to Roundtable Hold
                    1. After receiving the Chrysalids' Memento, Roderika relocates to Roundtable Hold. The Tarnished can meet her there to receive a Golden Seed as a reward.
                    2. She initially appears standing next to the fireplace in the main hall; exhausting her dialogue is required to obtain the Golden Seed.
                    3. If the Chrysalids' Memento was not given prior to her appearance at the Hold, the Golden Seed can still be acquired by returning to Stormhill Shack and looting it from the spot where she previously sat. Note: If Roderika was missed entirely at Stormhill Shack, selecting the "About Roderika" dialogue option with Smithing Master Hewg at Roundtable Hold allows the quest to continue normally. Note: Acquiring the Lake-Facing Cliffs Site of Grace early (by bypassing Margit) may auto-complete the quest up to this point without requiring the Chrysalids' Memento; this behavior may be unintended.
                Becoming a Spirit Tuner
                For Roderika to become Hewg's apprentice in spirit tuning, the Tarnished must:
                    1.  Speak to Smithing Master Hewg about Roderika.
                    2.  Inform Roderika of Hewg's remarks regarding her latent gift.
                    3.  Return to Hewg and persuade him that Roderika wishes to learn about spirit tuning.
                Once this sequence is completed, leaving and re-entering Roundtable Hold causes Roderika to relocate to the same room as Hewg, where she sits on the floor to the left, no longer wearing her crimson hood.
                From this point forward, Roderika functions as a Spirit Tuner, allowing the Tarnished to upgrade Spirit Ashes using Runes and Glovewort items. Note: If Roderika was missed at Stormhill Shack, selecting the "Please" dialogue option when first speaking to her at Roundtable Hold grants the Spirit Jellyfish Ashes.
            """
        },
        {
            "category": "quest",
            "query": "Create a quest for a side character who can be met exhausted and blindfolded near the Bridge of Sacrifice on the Weeping Peninsula, far from the seat of power. This is a quest about Irina, a gentle maiden with weak eyesight since birth, whose father Edgar serves as castellan of Castle Morne. During this quest, the Tarnished becomes a messenger between daughter and father, witnesses the tragedy of duty versus love, and ultimately confronts the consequences of loss and vengeance. Quest must contain three stages: initial encounter and letter delivery, returning to Irina after speaking with Edgar and resolving the castle conflict, then finding the final location where the quest concludes with a choice or inevitable outcome. Structure quest description as some steps using a list: describe player's actions (what they should do) and character's answer actions. For every stage of quest create minimum three steps (can be simple like finding person-speaking to him-receiving smth). Structure as lists of actions with headers, describing stages.",
            "reference": """
                Irina & Edgar Questline — Descriptive Overview
                Initial Encounter & Letter Delivery
                    1. The Tarnished encounters Irina near the Bridge of Sacrifice on the Weeping Peninsula, where she requests assistance in delivering a letter—"Irina's Letter\"—to her father, Edgar, castellan of Castle Morne at the southern tip of the peninsula.
                    2. Upon delivering the letter to Edgar at Castle Morne, he explains that he cannot abandon his duties to visit Irina, as he must protect the Grafted Blade Greatsword from being stolen by the rebellious Misbegotten. Note: If Edgar is killed at this stage, the questline progresses similarly, though certain dialogue and interactions are skipped.
                Return to Irina & The Iron Cleaver
                    1. Returning to Irina after speaking with Edgar reveals an Iron Cleaver embedded in the ground beside her—a sign that the Misbegotten have drawn near.
                    2. Defeating the Leonine Misbegotten boss at Castle Morne and retrieving the Grafted Blade Greatsword causes Edgar to thank the Tarnished and depart to reunite with his daughter.
                    3. The Tarnished can then find Edgar standing beside Irina at her original location near the Bridge of Sacrifice; speaking with him advances the questline.
                Final Encounter — Revenger's Shack
                    1. Later, the Tarnished encounters Edgar again at the Revenger's Shack in western Liurnia of the Lakes, where he has become an invader known as "Edgar the Revenger," consumed by grief and madness following Irina's death.
                    2. Defeating Edgar the Revenger concludes the questline. Note: Irina's fate is sealed regardless of the Tarnished's actions; by the time Edgar reaches her location after the defeat of the Leonine Misbegotten, she is already deceased, which ultimately drives his descent into vengeance.
                Additional Notes
                - The Iron Cleaver found near Irina serves as environmental storytelling, indicating the escalating threat from the Misbegotten even before the Tarnished confronts their leader.
                - Edgar's transformation into an invader is tied to the broader narrative of grief and the Flame of Frenzy, connecting this questline to other story threads in Liurnia and the Mountaintops of the Giants.
            """
        },
        {
            "category": "quest",
            "query": "Create a quest for a scholarly side character who guides the player toward uncovering a hidden historical conspiracy. This is a quest about a researcher who investigates ancient relics, interacts with multiple key NPCs to gather information, and gradually succumbs to a mysterious affliction tied to their findings. Quest must contain three to four stages: initial encounter and location change, uncovering a hidden memory and key item, delivering the item to advance a larger storyline, and witnessing the character's final decline. Structure the quest description using headers for each stage, followed by numbered lists. For every stage, include a minimum of three steps that alternate between player actions (exploration, dialogue, combat, item delivery) and character responses or quest progression. Keep the tone descriptive and the structure sequential, matching the length and pacing of a standard RPG companion quest.",
            "reference": """
                Sorcerer Rogier Questline — Descriptive Overview
                Initial Encounter & Stormveil Castle
                    1. The Tarnished discovers Sorcerer Rogier praying at an altar in the chapel within Stormveil Castle's northwest section, where he studies the lingering mysteries of the demigods. (This encounter must occur before defeating Godrick.)
                    2. After Godrick's defeat, Rogier relocates to the Roundtable Hold balcony overlooking the entrance hall, where he rewards the Tarnished with Rogier's Rapier +8 and the Ash of War: Glintblade Phalanx. Note: Progressing this questline is blocked if the Tarnished speaks to Ranni the Witch at the Three Sisters before completing Rogier's early objectives.
                The Tree Spirit & Ancient Memory
                    1. The quest directs the Tarnished beneath Stormveil Castle, where careful navigation through courtyards and scaffolding leads to Rogier's Letter and a hidden shortcut.
                    2. Descending to the roots below, the Tarnished defeats the Lesser Ulcerated Tree Spirit and discovers an ancient, rotting face marked by a bloodstain.
                    3. Activating the bloodstain reveals a spectral memory of the Night of the Black Knives; if the stain is invisible, the memory unlocks automatically after speaking to Rogier or defeating Godrick.
                The Black Knifeplot & Fia's Clue
                    1. Returning to Roundtable Hold, Rogier identifies the face as a sacred relic tied to the Black Knife conspiracy and directs the Tarnished to seek answers through intimate communion.
                    2. Allowing Fia to embrace the Tarnished and exhausting her dialogue regarding secrets and the Black Knifeplot yields the Knifeprint Clue, pointing to the Black Knife Catacombs.
                    3. Retrieving the Black Knifeprint from behind an illusory wall (guarded by a Black Knife Assassin) and delivering it to Rogier prompts him to send the Tarnished to Ranni the Witch in northwestern Liurnia for further guidance.
                Ranni's Vassalage & Rogier's Final Rest
                    1. Speaking with Ranni initially results in her dismissal of the Tarnished; reporting this to Rogier leads him to advise pledging as one of her vassals to gain her trust.
                    2. As Ranni's questline advances, Rogier warns the Tarnished that he is on the brink of a deep, unnatural slumber; resting at a Site of Grace confirms he has fallen into this sleep.
                    3. Further rest or story progression results in Rogier's death, leaving behind his Bell Bearing, attire, his rapier (if not already claimed), and a letter revealing the whereabouts of D's younger brother; his body and chair vanish upon subsequent visits.
            """
        },
        {
            "category": "quest",
            "query": "Create a quest for a minor merchant/NPC who is initially encountered as a petty criminal or thief. This is a quest about a pragmatic survivor who offers goods and information, gradually warms up to the player through purchases, and relocates to a more dangerous area as trust builds. The quest must intersect with a darker, parallel storyline, leading to a conditional invasion or tragic outcome. Structure the description using headers for each stage, followed by numbered lists. For every stage, include a minimum of three steps that alternate between player actions (dialogue, purchases, progression triggers) and character responses or consequences. Include conditional notes where player choices or quest timing alter outcomes, drops, or availability. Keep the tone descriptive and factual, matching the length and pacing of a standard RPG side quest.",
            "reference": """
                Blackguard Big Boggart Questline — Descriptive Overview
                Initial Encounter & The Stolen Necklace
                    1. The Tarnished first encounters Blackguard Big Boggart as the thief who stole Rya’s necklace; he shows no remorse and offers to sell it back for 1,000 Runes.
                    2. During this meeting, he warns the Tarnished that Rya "isn't right in the head," subtly hinting at her true nature and foreshadowing deeper lore.
                    3. The Tarnished may purchase the necklace or kill Boggart outright; choosing violence here permanently locks access to his merchant services and quest progression.
                Merchant Services & Boss Summon
                    1. If spared and Rya's questline is initiated, Boggart becomes available as a summon for the Magma Wyrm Makar boss fight.
                    2. Purchasing the necklace allows the Tarnished to buy Boiled Prawns from him; accepting his food establishes trust and advances his storyline, but also marks him for future danger. Note: Boggart's questline cannot progress if Rya's initial request is not accepted before the Tarnished reaches Volcano Manor.
                Relocation to Leyndell & Warning
                    1. After befriending him, Boggart relocates to the outer moat of Leyndell, Royal Capital, where he sets up a new stall selling Boiled Crabs.
                    2. He recounts his past imprisonment alongside the Dung Eater, describing "unspeakable acts" committed against corpses and urging the Tarnished to avoid him. Note: Arriving at this stage before progressing the Dung Eater questline may result in Boggart's premature death.
                Intersection with the Dung Eater & Invasion
                    1. When the Dung Eater’s questline reaches the "waiting in the outer moat" phase, exhausting Boggart’s dialogue reveals his suspicion that the Dung Eater is lurking nearby.
                    2. Reloading the area after this conversation triggers an invasion by the Dung Eater, targeting Boggart at his cooking pot.
                    3. Killing Boggart immediately after the dialogue but before reloading yields a seated corpse with standard drops, but no Seedbed Curse.
                Conclusion & Quest Rewards
                    1. Surviving the invasion or allowing it to resolve leaves Boggart’s body near the crab pot, containing his merchant drops and a Seedbed Curse essential for the Dung Eater questline. Note: If the Dung Eater has already invaded the moat before Boggart’s relocation, the Seedbed Curse cannot be obtained from his body.
            """
        },
        {
            "category": "dialogue",
            "query": "Create a short (3-4 sentences) monologue for a character Redmane Freyja, who is a former follower of Radahn, but is now a fervent follower of Miquella. This is a monologue where she intruduces herself and proceed with task: meeting the Tarnished who follows Miquella at this moment. This dialogue takes place after all events of world history. Do not structure monologue, write only Freyja's words.",
            "reference": """
                Ahh, Lady Leda spoke of you. You're that Tarnished, guided here by Kindly Miquella. Weren't we all. I am Freyja. I once fought alongside General Radahn. In battle, you can be sure I'll hold my own.
                Oh, another thing. Did you speak to our dour little friend? If you've yet to do so, have him give you a map of the crosses' whereabouts. They are Miquella's footprints.
                I urge you follow after Miquella.
            """
        },
        {
            "category": "dialogue",
            "query": "Create a short (3-4 sentences) monologue for a character Miriel, Pastor of Vows is the Liurna Church of Vow's steward, a huge silver tortoise wearing a mitre. This is a monologue where he intruduces himself and proceed with some story of the place he is located in: this is a church where Renalla and Radagon got married. The monologue takes place after all events of world history. Do not structure monologue, write only Miriel's words.",
            "reference": """
                You're Tarnished, aren't you? I Welcome you, to the Church of Vows. I am Miriel, steward of this sacred chamber. My apologies, for the unseemly state of affairs. Do you know the origin of this place? Who can blame you? The Shattering has caused us - all of us - to lose sight of something very dear. It is here, at the Church of Vows, that the great houses of the Erdtree and the Moon were joined. And so, our church holds in view the monuments of both houses. The Erdtree of the Capital, and the Academy of Raya Lucaria.
            """
        },
        {
            "category": "dialogue",
            "query": "Create a short (3 phrases) monologue for a Rennala after all worlds events and even after player defeated her. This are phrases that starts Rebirth: Rennala gives access to the Rebirth function, allowing the player to reallocate their stats. Do not structure monologue, write only Rennala's words.",
            "reference": """
                Ah, it seemeth thou'rt a sweeting rather fair.
                And if thou wert born anew once more, thou wouldst be fairer still.
                Ye will be countless born, forever.
            """
        },
        {
            "category": "dialogue",
            "query": "Create a monologue about three paragraphs with 3-4 sentences each. This is monologue of Sorcerer Rogier, who is a Tarnished NPC in Elden Ring. Rogier is a spellblade, and seeks to discover the truth behind the Night of the Black Knives. He visits some ruins and other places and investigates information about this event. This certain monologue should be about Runni the Witch and her participation in the Night of the Black Knives, but all Rogier's words are only suggestions, he is not sure about this. This monologue happens after all events and all world's history, so it is modern suggestions about past events. Do not structure text, write only Rogier's words without any comments.",
            "reference": """
                Now, I have a fairly good idea who performed the rite upon the blade. The person who orchestrated the Night of the Black Knives. Lunar Princess Ranni. One of the children born to King Consort Radagon and his first wife, Rennala. Demigod and sister to General Radahn and Praetor Rykard. Her's was the name I discovered in the imprint. Truly, you have my thanks. But, if I might be so bold, I would also like to ask something more of you. If Ranni truly is the one who plotted that fateful night, then she should bear the cursemark of Destined Death somewhere upon her flesh. I would like you to procure it for me. And then all will be laid bare. I will have the answers I have sought for so long.
                I have some idea of Ranni's potential whereabouts. There's a manor to the north of the Academy of Raya Lucaria. It is the familial home of the Carian royals from whom Ranni descends. There's been talk of the old royals' vassals gathering there in recent years. Ranni's whereabouts since the Shattering are a well-kept secret. She hasn't been seen even once. But I suspect she might have returned to the manor in which she was born...
                I see... When Ranni shed her flesh, she shed the cursemark, too. You know, not everyone would trust such a tale... But, if she in her current form is nothing more than the living doll you profess... Then perhaps it's true after all.
            """
        },
        {
            "category": "dialogue",
            "query": "Create a monologue about 8 short phrases, in which Melina very abstractly and with no facts answers about her story. This happens after all world history and all events, and she serves player's Tarnished as a maiden, but she is not a maiden so she apologizes. Use short phrases as Melina do not want to talk about her past but follows her own purposes collaborating with player. Answer only with Melina's words, with no other commentaries or structures. It has to be a monologue, so no player's words.",
            "reference": """
                Me?
                I'm searching.
                For my purpose, given to me by my mother inside the Erdtree, long ago.
                For the reason that I yet live, burned and bodiless.
                There is something for which I must apologise.
                I've acted the Finger Maiden, yet can offer no guidance.
                I am no maiden.
                My purpose...was long ago lost.
            """
        },

    ]
}