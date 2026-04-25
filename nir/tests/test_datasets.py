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