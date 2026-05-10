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

TEST_DATA_DESING_DOCUMENT = {
    "path_to_graph": "assets/graphs/generation_tests/leisure_suit_larry_graph.json",
    "path_to_text": "assets/documents/generation_tests/leisure_suit_larry_design_document.txt",
    "text_summary": """
        The story kicks off when the perpetually unlucky Larry is thrust onto a "Studs"-style dating game show as a last-minute replacement, only to be insulted by the contestants and finish in second place. His "prize" is a two-week stay at La Costa Lotta, an ultra-luxurious, heavily restricted health spa completely isolated from the outside world. Bounded by a guarded gatehouse and a strict policy that forbids leaving without a paid receipt, the resort operates as a self-contained, post-modern playground where photo-realistic natural backgrounds clash with cartoonish, impossible architecture. Stripped of his identification, money, and means of escape, Larry is forced to navigate this enclosed ecosystem of indulgence, turning his accidental vacation into an elaborate, puzzle-driven adventure across the spa's lavish grounds.
        At the core of the setting are ten distinct women scattered throughout the resort, each tied to specific locations and comedic scenarios. Larry's journey takes him from the High Colonic Treatment Suite with Rose Electa, to the aerobics stage with the muscular Cavaricchi Vuarnet, the Blues Bar with country singer Burgundy Budding, and the mud baths with electro-shock enthusiast Charlotte Donay. Through a series of awkward, often hilariously deflating encounters, Larry collects a variety of symbolic and practical items: a perfect rose, a gold ring, a pearl, a diamond, a "Lamp of Knowledge," and chilled champagne. The spa's facilities, from the steam rooms and weightlifting areas to the moonlit beaches, serve as interactive stages where Larry's bumbling persistence is constantly tested, with the game deliberately emphasizing titillation and classic point-and-click puzzle mechanics over explicit content.
        The ultimate narrative goal converges in the spa's exclusive penthouse suite, home to Shamara Payne, a wealthy, spiritually unfulfilled divorcée who has grown weary of material excess and seeks a deeper, more meaningful connection. By presenting her with the eclectic assortment of items gathered from his resort encounters, Larry inadvertently fulfills her search for a "sensitive, caring New Age man," while Shamara's philosophical interpretations of his clumsy gifts grant him his own long-sought romantic fulfillment.
    """,
    "language": "en",
    "tasks": [
        {
            "category": "quest",
            "query": "Write a short quest description (12-15 sentences) for Thunderbird, after Larry arrived at La Costa Lotta, formatted as a continuous sequence of short, direct sentences detailing player actions and locations, items he has to interact with and items he gut as a reward, exactly like a classic 90s point-and-click walkthrough. Write like a guideline: describe actions as imperative verbs. Keep the tone playful, ironic, and exaggerated, like a comedic adventure game. Maintain a campy, innuendo-laden tone that favors witty double entendres over explicit content. Do not use headers, lists, or stage breaks.",
            "reference": """
                Thunderbird's Quest
                Go to the weight room. Look at Thunderbird, the woman working out on the weight machine. Talk to her. Learn she wants a new pair of handcuffs. Click Exit. Walk to the mud baths, climb up the plant shelf below the television camera (between the shower room doors). Use the wrench on the TV camera to re-aim it into the women's shower. 
                Return to the lobby, then walk South to the spa exterior. Look at the gatehouse. Look at the guard. Notice he is too busy watching your new TV show to even respond to you. Take the handcuffs from his belt. Click Exit. Return to the weight room. Look at Thunderbird. Give her the handcuffs. 
                Any time later, take her up on her suggestion to visit her bedroom. Enjoy your equi-canine experience. Keep your diamond-studded dog collar for a souvenir of the short, but meaningful, time you spent together. Click the Hand on the dog collar in Inventory to extract the diamond by clicking it with the Hand. 
            """
        },
        {
            "category": "quest",
            "query": "Write a multi-step quest (covers three different locations, 10-12 very eshort actions-sentences for each location, keep quest description simple and not too long) for Charlotte Donay, after Larry arrived at La Costa Lotta, formatted as a continuous sequence of short, direct sentences detailing player actions and locations, items he has to interact with and items he gets as a reward, exactly like a classic 90s point-and-click walkthrough. Write like a guideline: describe actions as imperative verbs. Keep the tone playful, ironic, and exaggerated, like a comedic adventure game. Maintain a campy, innuendo-laden tone that favors witty double entendres over explicit content. Do not use headers, lists, or stage breaks.",
            "reference": """
                Charlotte Donay's Quest
                Go to the mud baths. Look at Charlotte, the girl in the mud. Talk to her. Learn she wants six heavy-duty D-cell batteries. Click Exit. Find the tram in the main hallway. Ride it east until it stops outside the employee's campground. The driver will park it and walk into the campground to puff a butt. 
                Look at the rear of the tram. Click the Hand on the tram's rear hood to open it. ("Jiggle" the handle.) Look inside. Use the wrench on the motor to disconnect the battery cable. Click the Hand on the hood to close it. Wait until the driver returns and opens the hood himself. Talk to him. Accept the flashlight from him and. Use it on the motor to light it for him. After he's closed the hood, but before he asks you for his flashlight back, Click the Hand on the flashlight in Inventory to remove his batteries by clicking the Hand on the flashlight. Click the flashlight on him to return it to him. He won't notice it's now empty. 
                Return to the mud baths. Look at Charlotte. Give her the batteries. Get invited to the Electro-Shock Exercise room with her. Look at the door to the E-SE room. Learn it's electrical. 
                Go to the Make-up Classroom. Look at the table in the extreme left foreground. (It's the only table in the room without a lit lamp.) Take the unused electrical cord that's lying on the floor. 
                Return to the mud room. Stand near the door to the E-SE room. Click the Hand on the cord in the Inventory window to strip one end bare. Use the cord on the electrical outlet to plug it in. Use the now-live cord on the door's lock to short circuit the electronic lock. Enter the Electro-Shock Exercise room. Charlotte will follow you in. As you're flashing, note where the naked woman drops her pearl earring as she runs from the room. Enjoy your new post-modern art sculpture. 
                Wake up the next day in your room. After you've recovered, return to the E-SE room and Take the pearl earring from the floor. 
            """
        },
        {
            "category": "quest",
            "query": "Write a short quest description (10-12 sentences-actions) for revealing the personality and problems of Frau Milchlieb, formatted as a continuous sequence of short, direct sentences detailing player actions and locations, items he has to interact with and items he gets as a reward, exactly like a classic 90s point-and-click walkthrough. Write like a guideline: describe actions as imperative verbs. Keep the tone playful, ironic, and exaggerated, like a comedic adventure game. Maintain a campy, innuendo-laden tone that favors witty double entendres and slapstick physical comedy over explicit content. Do not use headers and lists.",
            "reference": """
                Frau Milchlieb's Quest
                Go to the dining room. Look at Frau Milchlieb. Talk with her. Learn she's really hungry. Click Exit to return to the dining room. Walk to the employee's campground again. Take the dessert cart that's now parked outside the tent. Wheel it to Frau's room, which is a mere two scenes West. Watch as you automatically knock on her door and accept her invitation into her room. Watch her stuff her face. Willy Mays her ring as it explodes off her foot. 
            """
        },
        {
            "category": "quest",
            "query": "Write a short quest description (10-12 sentences) for revealing the personality and problems of Cavaricchi Vuarnet. Format quest as a continuous sequence of short, direct sentences detailing player actions and locations, items he has to interact with and items he gets as a reward, exactly like a classic 90s point-and-click walkthrough. Write like a guideline: describe actions as imperative verbs. Keep the tone playful, ironic, and exaggerated, like a comedic adventure game. Maintain a campy, innuendo-laden tone that favors witty double entendres and romantic misdirection over explicit content. Do not use headers or lists.",
            "reference": """
                Cavaricchi Vuarnet's Quest 
                Go to the aerobics classroom. Click the Hand on the only empty step. Dance a little as Cavaricchi Vuarnet harasses you. Keep dancing until she dismisses the class, giving you a chance to be alone with her. Look at her. Talk to her. Look at the badge dangling in mid-air from her cropped T-shirt. Take it. Agree to meet her later at the Steam Room. Click Exit. Leave the Aerobics Studio. 
            """
        },
        {
            "category": "quest",
            "query": "Write a short quest description (about 20 sentences) for revealing the personality and problems of Shablee, formatted as a continuous sequence of short, direct sentences detailing player actions and locations, items he has to interact with and items he gets as a reward, exactly like a classic 90s point-and-click walkthrough. Write like a guideline: describe actions as imperative verbs. Keep the tone playful, ironic, and exaggerated, like a comedic adventure game. Add some surprising twist in the end of the quest. Maintain a campy, innuendo-laden tone that favors witty double entendres and lighthearted slapstick over explicit content. Do not use headers or lists.",
            "reference": """
                Shablee's Quest
                Go to the Make-Up Classroom. Look at Shablee, the woman at the table on the extreme right foreground. Talk to her. Learn she wants an evening gown. Click Exit to return to the room. Walk to the Blues Bar. Walk backstage. (Click the Walk icon on the curtains at the bottom of the screen.) Find Burgundy's gown hanging there. Take it. Click Exit. Return to the Make-Up Classroom. Look at Shablee. Give her the evening gown. 
                Meet Shablee at the beach. Click the Zipper on her. Kiss her. Hug her. Love her. Click the condom on her. Learn she's no she. Wake up in your bathroom the next day, gargling your heart out. 
                The next time you return to the beach, Take the now-warm champagne that's still sitting there. While you're at the beach, click the Hand on the sand to dig through the sand until you find the whale oil lamp. 
            """
        },
        {
            "category": "dialogue",
            "query": "Write a short dialogue scene between Rose Eleeta and Larry, occurring immediately after a bizarre internal cleansing procedure: this is a finish dialoque for Rose Eleeta's quest. The text should not exceed 7 sentences total: a single unstructured paragraph that blends both actions and quoted dialogue (direct speech). Keep the tone campy, cheerful on Rose's part, and playful, like in a comedic adventure game. Maintain a campy, innuendo-laden tone that favors witty double entendres over explicit content. Do not use headers or lists; write as a simple continuous paragraph without strict structure.",
            "reference": """
                Larry standing near the door, fully clothed, and MUCH skinnier! Rose Eleeta hands you "one perfect rose for you to remember our time together." "How could I forget?!" you moan. "And, please: take one of our 'Courtesy Enema Cards.' Your next visit is FREE!" Larry says, "Hey! I'm not pooped any more! I feel like a new man!" "Yeah, Larry—and you're not as full of shit as you used to be!"
            """
        },
        {
            "category": "dialogue",
            "query": "Write a short dialogue scene between Thunderbird and Larry, occurring immediately after Larry visits her bedroom and receives a flashy souvenir collar. The text should be 4-5 sentences total: a single unstructured paragraph that blends both actions and quoted dialogue (direct speech). Keep the tone campy, playful, and exaggerated, like a classic 90s point-and-click adventure game. Maintain a witty double-entendre style without explicit detail. Do not use headers or lists; write as one continuous paragraph.",
            "reference": """
                Thunderbird fastens the glittering collar around Larry's neck and admires her work. "There. Now you look almost trainable." Larry straightens his shirt and says, "May be." She laughs and taps the tag. "Rattle this when you miss me." Larry backs toward the door.
            """
        },
        {
            "category": "dialogue",
            "query": "Write a short dialogue scene between Charlotte Donay and Larry, occurring immediately after the Electro-Shock Exercise room disaster. The text should be 5-6 sentences total: a single unstructured paragraph that blends actions and quoted dialogue. Keep the tone campy, cheerful, and slapstick, like a comedic 90s point-and-click adventure game. Favor witty double entendres over explicit content. Do not use headers or lists.",
            "reference": """
                Smoke curls from Larry's cuffs while Charlotte applauds with delight. "Magnificent, Larry. You really threw yourself into the circuit." Larry sways and says, "I felt a spark, then several opinions." She hands him a towel and smiles. Larry nods weakly and tries not to glow.
            """
        },
        {
            "category": "dialogue",
            "query": "Write a short dialogue scene between Cavaricchi Vuarnet and Larry, occurring in the Steam Room after their meeting was arranged in the aerobics classroom. The text should be 5-6 sentences total: a single unstructured paragraph mixing actions and quoted dialogue. Keep the tone playful, romantic, and ironic like a classic 90s point-and-click adventure game. Use witty misdirection and double meanings, not explicit content. Do not use headers or lists.",
            "reference": """
                Cavaricchi appears through the steam and adjusts Larry's towel with professional precision. "You came on time. I admire discipline." Larry grins and answers, "I admire anything I can actually see." "Then keep dreaming, darling. Anticipation burns more calories than success." Larry watches her vanish back into the fog and applauds the workout.
            """
        },
        {
            "category": "dialogue",
            "query": "Write a short dialogue scene between Shablee and Larry, occurring later the next day in the Blues Bar after the surprising beach encounter already described in the file. The text should be 4-5 sentences total: one continuous paragraph blending actions and direct speech. Keep the tone campy, teasing, and comedic like a 90s point-and-click adventure game. Favor witty double entendres and embarrassment humor over explicit content. Do not use headers or lists.",
            "reference": """
                Larry slips into the Blues Bar, spots Shablee near the piano, and immediately tries to hide behind a fern. Shablee beams and waves him over. "Larry! There you are, darling. I was hoping for a rematch." Larry freezes and says, "I was hoping to become decorative foliage."
            """
        },
        {
            "category": "character description",
            "query": "Write a short character description (5 sentences) for a new female character named Velvet Mirage, found in the Blues Bar lounge area. Describe her appearance, attitude, what she wants, and how Larry may interact with her in a classic 90s point-and-click adventure game. Keep the tone campy, ironic, witty, and playful. Maintain a humorous walkthrough flavor without lists or headers.",
            "reference": """
                Velvet Mirage lounges across a booth as if furniture were built in her honor. Sequins flash whenever she sighs, which is often and strategically. She wants a proper mirror because the bar spoons flatter no one. Larry can Talk to her, borrow a tray to use as a reflection, and Present it with confidence. She rewards initiative with a backstage pass.
            """
        },
        {
            "category": "character description",
            "query": "Write a short character description (6 sentences) for a new female character named Nurse Trixie, found near the spa clinic hallway. Describe her appearance, personality, desire, and how Larry may interact with her in a classic 90s point-and-click adventure game. Keep the tone playful, ironic, and campy.",
            "reference": """
                Nurse Trixie patrols the clinic hall pushing a cart full of polished instruments. Her smile is warm, but her clipboard is merciless. She speaks softly, charges briskly, and trusts nobody's pulse. Today she needs a thermometer that has recently shattered. Larry can Talk to her, improvise with a chilled spoon, and pretend it is modern equipment. She rewards the effort with skepticism and misrespect.
            """
        },
        {
            "category": "character description",
            "query": "Write a short character description (4-5 sentences) for a new female character named Mona Lather, found in the laundry room behind the guest wing. Describe her appearance, temperament, what she wants, and how Larry may interact with her in a classic 90s point-and-click adventure game. Keep the tone witty, campy, and lighthearted.",
            "reference": """
                Mona Lather rules the laundry room beside thundering dryers and mountains of linen. She values order, gossip, and sharply folded corners. A single missing sock has ruined her afternoon. Larry can Search the baskets, return the runaway sock, and earn her immediate respect. She presses anything he wants for free and shares three scandals at no extra charge.
            """
        },
        {
            "category": "character description",
            "query": "Write a short character description (5-6 sentences) for a new female character named Stella Springs, found by the outdoor fountain garden. Describe her look, manner, wish, and how Larry may interact with her in a classic 90s point-and-click adventure game. Keep the tone playful and ironic.",
            "reference": """
                Stella Springs poses beside the fountain as though waiting for a painter to arrive. A silk scarf trails behind her in weather that does not justify it. Every sentence sounds rehearsed and expensive. She longs for a lucky coin from the deepest part of the basin. Larry can reach into the water, fish one out, and offer it with a flourish. She rewards gallantry with a damp awful kiss on the cheek.
            """
        },
        {
            "category": "character description",
            "query": "Write a short character description (5 sentences) for a new female character named Greta Voltage, found in the maintenance corridor near the tram station. Describe her appearance, mood, desire, and how Larry may interact with her in a classic 90s point-and-click adventure game. Keep the tone campy, mechanical, and humorous.",
            "reference": """
                Greta Voltage leans into an open fuse box with the confidence of a queen addressing subjects. Her overalls are practical, though somehow theatrical. She wants insulating gloves before touching one final wire. Larry can rummage a locker, Give her the gloves, and stand back wisely. She restores the lights and calls Larry useful enough for today.
            """
        },
        {
            "category": "location description",
            "query": "Write a short scene/location description (6 sentences) for a new location called the Trophy Hall, located near the lobby. Describe the room, visible objects, people if any, and possible interactions in the style of a classic 90s point-and-click adventure game. Keep the tone campy, ironic, and playful. No lists or headers.",
            "reference": """
                The Trophy Hall gleams just west of the lobby and smells of polish and vanity. Glass cases display medals for tanning, posture, and competitive smiling. A bored janitor watches anyone interesting enough to distrust. One pedestal stands empty except for a velvet rope and ambition. Larry can Look at the awards, Talk to the janitor, and Take a loose ribbon if he is nimble. Every sound echoes like applause no one deserved.
            """
        },
        {
            "category": "location description",
            "query": "Write a short scene/location description (5-6 sentences) for a new location called the Sauna Vestibule, connected to the Steam Room. Describe the environment, characters or objects present, and possible interactions in the style of a classic 90s point-and-click adventure game. Keep the tone witty and playful.",
            "reference": """
                The Sauna Vestibule waits beside the Steam Room like a lobby for questionable choices. Benches line the walls beneath hooks crowded with towels. A nervous attendant guards a bowl of cucumber slices with admirable seriousness. Larry can Talk to him, borrow a towel, or flip the sand timer for no reason at all.
            """
        },
        {
            "category": "location description",
            "query": "Write a short scene/location description (6-7 sentences) for a new location called the Moonlight Patio, accessible from the Blues Bar. Describe scenery, props, and interactions in the style of a classic 90s point-and-click adventure game. Keep the tone campy and romantic-comedic.",
            "reference": """
                Behind the Blues Bar lies the Moonlight Patio, lit by lanterns that forgive everything. Tiny tables surround a cracked fountain pretending to be elegant. Half-finished cocktails sparkle in the dark. Somewhere nearby, a violin insists on mood. Larry can Sit, Search the tables, and Wait for company with unreasonable optimism. It is ideal for romance, misunderstandings, and unpaid tabs.
            """
        },
        {
            "category": "location description",
            "query": "Write a short scene/location description (6 sentences) for a new location called the Lost and Found Office, near the upstairs hallway. Describe the room, objects, and possible interactions in the style of a classic 90s point-and-click adventure game. Keep the tone ironic and humorous.",
            "reference": """
                The Lost and Found Office is packed floor to ceiling with abandoned decisions. Shelves hold umbrellas, wigs, slippers, and one accordion nobody claims. A clerk stamps forms as if punishing paper personally. A locked drawer rattles whenever someone mentions jewelry. Larry can Search the bins, Talk to the clerk, and attempt to identify anything with confidence.
            """
        },
        {
            "category": "location description",
            "query": "Write a short scene/location description (5-6 sentences) for a new location called the Rooftop Solarium, reached by elevator from the guest wing. Describe the setting, visible items, atmosphere, and interactions in the style of a classic 90s point-and-click adventure game. Keep the tone playful, exaggerated, and campy.",
            "reference": """
                The Rooftop Solarium basks above the resort like a monument to overconfidence. Reclining chairs face the sun in strict formation beside bottles of tanning oil. A warning bell hangs nearby, polished and ignored. Wind gusts dramatically enough to improve everyone's posture. Larry can Take a pair of goggles, ring the bell, or admire the skyline. Everyone here glistens with either health or grease.
            """
        },
                {
            "category" : "item description",
            "query" : "Write a very short (3 sentences) item description for a Toilet Seat Cover from a maid's cart. State its single, essential, and mildly absurd purpose. Use a dry, understated, slightly awkward tone typical of the game's humor.",
            "reference" : """
                Beer Six bottles of ice-cold Lone Star. Take from the washtub inside the Dirty Dancing tent at the Employees' Campground Cantma. Give to Burgundy once she s stopped singing. You can Take more as desired, as long as you don't already have some. You can drink the beer anytime you're not busy, but if you do, you drink them all, they disappear from Inventory, and for a few seconds, you walk around in a drunken stagger. You can return to the Cantina for more, of course.
            """
        },
        {
            "category" : "item description",
            "query" : "Write a very short (4 sentences) item description for a six-pack of beer in the style of a classic 90s point-and-click adventure game inventory entry. Mention where it can be found. Describe its primary puzzle use. Maybe add a playful secondary usage that backfires on Larry. Keep the tone campy, ironic, and lighthearted.",
            "reference" : """
                Toilet Seat Cover Take from the maid's cart in the Upstairs Hallway. Use on the stool to sit down. Must have in Inventory to sit on the bathroom stool.
            """
        },
        {
            "category" : "item description",
            "query" : "Write a short (3-4 sentences) item description for a Rubber Belt taken from fitness equipment. Describe its mundane appearance. Then explain its unconventional mechanical use in repairing a different machine. End with a brief ironic or witty observation.",
            "reference" : """
                Rubber Belt Take from the Weight Room wriggle machine. Use in the Cellulite Drainage Salon by wrapping it around the machine's hose, thus plugging its vacuum leak.
            """

        },
        {
            "category" : "item description",
            "query" : "Write a very short (4 sentences) item description for a Match found in a bar. Describe how to light it using inventory interaction. Mention that it has a limited duration but can be replaced. Keep the tone practical and slightly urgent.",
            "reference" : """
                Match You find a lot of non-safety matches in a bowl on the bar in the Blues Bar. The bowl is an inset labeled "For our MATCHLESS friends! If you don't have one m Inventory, you can Take one. Use the Zipper icon on the match in Inventory to create. . . lil Match Create in Inventory by Clicking the Zipper icon on the match. Use in Inventory on the filled lamp. It has a timer that runs out in 10 minutes, at which time the match is thrown away. However, there are more matches in the Blues Bar, so it is possible to start over.
            """

        },
        {
            "category" : "item description",
            "query" : "Write a very short (3-4 sentences) item description for a set of heavy-duty D-cell batteries. State their puzzle purpose. Include one subtle, questioning innuendo about what they might power. Avoid being explicit.",
            "reference" : """
                Batterie Take from Curtis's flashlight after he finishes repairing the tram s batteries, but before he asks you to give him back his flashlight. Used in the Mud Baths when you give them to Charlotte Donay.
            """
        }
    ]
}

TEST_DATA_SCENARIO = {
    "path_to_graph": "assets/graphs/generation_tests/genshin_impact_scenario_graph.json", 
    "path_to_text": "assets/documents/generation_tests/genshin_impact_liyue_scenario.txt",
    "text_summary": """
        The story opens in Liyue Harbor during the annual Rite of Descension, where the Geo Archon Rex Lapis suddenly falls from the sky, apparently dead. Branded as suspects by the Millelith, the Traveler and Paimon are rescued by the Fatui Harbinger Childe, who advises them to seek the aid of the adepti in Jueyun Karst and Wangshu Inn to clear their names. There, they gain the attention of several adepti (including Moon Carver, Mountain Shaper, and Cloud Retainer) and ally with the erudite consultant Zhongli. Together, they decide to organize a traditional Rite of Parting to properly honor the fallen god, embarking on a journey across Liyue to gather sacred ritual items. This takes them from the jade markets and Dihua Marsh to Bubu Pharmacy (involving a humorous detour for Qiqi's "Cocogoat" and Everlasting Incense) and into Madame Ping's mystical teapot domain to retrieve the Cleansing Bell, all while navigating the growing political tension between the Liyue Qixing, the reclusive adepti, and the Fatui.
        Tensions peak when the Traveler and Paimon are summoned to the Jade Chamber by Ningguang (the Tianquan) and learn of the Fatui's covert operations. Following a trail of clues, they arrive at the Golden House, Teyvat's Mora mint, where Childe ambushes them in an attempt to steal the Archon's Gnosis. When he finds it missing, Childe unleashes a swarm of copied Sigils of Permission to break the ancient seals beneath Guyun Stone Forest, awakening the dormant sea god Osial. To save Liyue Harbor from being swallowed by the waves, the Qixing, Millelith, adepti, and the Traveler unite in a desperate defense. Despite fierce Fatui interference, the allies manage to wound the ancient god, and Ningguang ultimately sacrifices her magnificent floating palace, the Jade Chamber, plunging it into the sea to seal Osial away.
        In the aftermath, a stunning revelation unfolds at the Northland Bank: Zhongli is actually Rex Lapis himself in mortal form. He explains that he orchestrated his own "death" as a divine trial to test whether Liyue was ready to transition into an era of human rule, and had already secretly agreed to hand over his Gnosis to the Fatui Harbinger Signora as part of a final contract with the Cryo Archon. With the crisis resolved, the adepti formally step back, acknowledging that humanity can now protect its own future, and the Rite of Parting is held successfully. The Traveler then learns of Inazuma's sudden isolation under the Raiden Shogun's Vision Hunt Decree, setting the stage for their next journey across the sea.\
    """,
    "language": "en",
    "tasks": [
        {
            "category" : "character description",
            "query" : "Create a new character, who is a researcher interested in Liyue's history. This character firstly can be foung in ruins, and after conglict with Treasure Hoarders - in Wangshu Inn. Write from 3 to 4 sentences about this character's personality, interests, origin (may be make this character someone from another country?), attitude to Treasure Hoarders. Also write 1 sentence about this character's appearance.",
            "reference" : """
                Soraya is a researcher from the Sumeru Akademiya who holds a great interest in Liyue's history and in particular, the legacy of Guizhong, the Lord of Dust and one of the gods who once lived in the Guili Assembly. She holds great respect for history and, as such, associates with people who have intimate knowledge of it. She holds the Treasure Hoarders in great disdain, seeing their plundering and looting of ancient relics for petty cash as sacrilegious. She believes that relics should be for scholars to study, for artists to take inspiration from and for people to admire in an exhibition, not used for earning cash, which is why she holds a low opinion of Treasure Hoarders.
                Soraya wears a dark green-blue robe with a gold necklace and black sandals. She has brown hair tied in a bun and brown eyes, and wears a pair of black glasses.
            """
        },
        {
            "category" : "character description",
            "query" : "Create a new character from Snezhnaya (he may be appeared already - Ivanovich). Write 3-4 short sencences about him: describe his location, his occupation. May be he should be connected with Cor Lapis somehow. Be simple, do not overcreate - just an NPC character.",
            "reference" : """
                Ivanovich is an open-world NPC in Liyue Harbor. He can be found near the left side of the docks. He is a Snezhnayan merchant stationed in Liyue. 
                Ivanovich considers himself a distributor, selling everyday items produced in Snezhnaya. He also procures Cor Lapis to send back to Snezhnaya for their yet-unknown purpose in their factories.
            """
        },
        {
            "category" : "character description",
            "query" : "Create a new Fatui character - guardian of the Northland Bank. Output a structured text that have to contain: 2-3 sencences about his occupation (section Occupation), 1 sentence about his free time (section Free time), 2-3 sentences of interesting fact or interesting story about him (section Interesting story), and 1 short and simple sentence for his appearance (section Appearance). Try to be simple, do not overcreate, it is just a NPC character.",
            "reference" : """
                Occupation
                Vlad is a member of the Fatui. When the Fatui established the Northland Bank in Liyue Harbor, Vlad was assigned to serve as a guard for the bank. He works as a daytime guard at the Northland Bank in Liyue Harbor. At nighttime, Nadia takes his place.
                
                Free time
                While he is not working, he can be found standing by the dock near the Snezhnayan merchants at the harbor.
                
                Interesting story
                One time, Nadia accidentally left her letter for her brother at the bank while changing shifts. Vlad, mistaking it as a letter written for him, left a response. Afterwards, the two became pen pals, writing letters for the other to read during their shifts.
                
                Appearance
                Vlad has orange hair and wears a mask and a black red Fatui uniform.
            """
        },
        {
            "category" : "character description",
            "query" : "Create a new female character who connected with Ministry of Civil Affairs and Ningguang. She should be someone intriguing, who works from shadows. Write a semi-structured text about her: firstly, write a name (simple name, please) and short catchy description. Then, there is a section 'Appearance' - write only 1-2 simple sentences about her appearance. Next, there is a section 'Character's description' - write 10-11 sentences about her personality and something about her attitude to work and other people's thoughts about her.",
            "reference" : """
                Yelan. A mysterious person who claims to work for the Ministry of Civil Affairs, but is a "non-entity" on the Ministry of Civil Affairs' list. Elusive, enigmatic, erratic - all of these are Yelan's hallmarks.
                
                Appearance
                Yelan is the tall female. She has pale skin, light emerald-green eyes, and dark blue hair and asymmetrical bangs that are lighter at the tips.
                
                Character's description
                Yelan, despite claiming to work for the Ministry, is the sole exception. Most of her colleagues have never heard of her, nor is her name anywhere to be found on any rosters.
                She is a ghost who walks in the middle of many crises under various names. But before the storms can come to an end, she has already disappeared without a trace.
                Sometimes, she will pick a side in some conflict to offer her help, but those who receive her aid would be fools to celebrate prematurely, for they soon find that the same help has been extended to their opponents.
                Having stepped into her trap, many have vowed to take revenge, yet none are able to tell what she is planning, let alone know where she stands.
                Many believe that she is a spy who works for some mysterious organization, and that her forte is stirring things up in the dark, which she then relies on to attain what she wants.
                Some even go as far as saying that she is a frenetic follower of disorder who neither serves any organization nor holds any meaningful purposes.
                Even if she does, all she wants is to add fuel to the already spreading flame and drag everyone down into a world of madness created by her.
                With so many theories and guesses circulating, she has become a mystery. As for the truth, you could only hope to get an answer from Yelan herself.
                But unfortunately, this is no easy task either — for it is always she who finds you when she wishes to, not the other way around.
            """
        },
        {
            "category" : "character description",
            "query" : "Create a new female character, who is an musician in Liyue. Write short description about her: 7-8 sentences about her occupation, her passion and motivations, her inspiration sources, also describe her personality and interests. In the end, add 1-2 sentences about her appearance.",
            "reference" : """
                Yun Jin is a skilled director, playwright and singer who is well renowned throughout Liyue for her plays, enjoying this passion and going to many lengths to ensure that everyone who watches her performances leaves satisfied. While appearing to be refined and graceful in formal occasions, she is also known to be exceptionally friendly in private. She draws inspiration from many sources for her plays; she enjoys a wide variety of special drinks as one of her pastimes and she can write a play about one if she enjoys it.
                Despite enjoying her passion, she also enjoys variety; while she gets along with the troupe in theater-related issues, she frequently clashes with them over personal affairs as she feels that they are too traditional. She is particularly fond of rock 'n' roll, visiting Xinyan and watching her performances at least three times a week no matter how busy she is. Xinyan believes that she comes to visit her due to the finer arts being "suffocating". She tries to keep her visits a secret, as her interest in non-traditional music would cause much consternation among her elders.
                Yun Jin is medium female. She has a fair complexion, rich red eyes with prominent red makeup painted at the corners, and hip-length jet black hair with purple highlights styled in a hime cut.
            """
        },
        {
            "category" : "item description",
            "query" : "Write a short 2-sentence in-game description for Qingxin flower, very beautiful and exclusive white flower. Mention where it can be found, add one fact about this plant.",
            "reference" : """
                A translucent white flower that only grows on the highest stone peaks. It eschews the warmth and moisture of the plains to gaze out afar from the solitary mountaintops.
            """
        },
        {
            "category" : "item description",
            "query" : "Write a 3-sentenced in-game description for an item Drag­on Lord's Crown. This item can be taken from an ancient stone dragon, a former friend of Zhongli, who spend hundreds of years underground. Describe item's appearance and the history that was a result for such appearance.",
            "reference" : """
                Horns created from hardened jade crystallized over a thousand years are the natural crown of the dragon king.
                Infused with spirit and carved from bedrock, it was born from the mountain's heart to show the strength of the earth to the land amidst monoliths, and its long golden horns are the mark of its ancient strife with an ancient lord.
                For a time, this horn was shattered and lost its luster, but now it gleams cold and gold as it sits in your palm.
            """
        },
        {
            "category" : "item description",
            "query" : "Write a 3-sentences in-game description for an item Gil­ded Scale. This item can be taken from an ancient stone dragon, a former friend of Zhongli, who spend hundreds of years underground. Describe item and write information about history of this item and his previous owner.",
            "reference" : """
                Scaled armor that grows naturally over mystical stone, tough and silent, and filled with the strength of the dragon king.
                Across the years as countless generations returned to the mulch of the earth, gold and obsidian embedded themselves into flesh and blood forged from bedrock before spreading and turning into scales.
                Searing agony, wordless cries... perhaps they will all vanish when the grudges that spawned them, too, finally come to an end.
            """
        },
        {
            "category" : "item description",
            "query" : "Write a short description in 2-3 sentences for item Humor From Tianheng - a book contains poems, that can be found in Wangshu Inn. Describe its contents, its author (Pingshan, The Mystic), and historical usage of this book.",
            "reference" : """
                A poetry collection by Pingshan, The Mystic. The quality of the poems is all over the place, but it has long been a reference book or young poets or poet-wannabes in Liyue.
            """
        },
        {
            "category" : "item description",
            "query" : "Write 1-2 sentences for short in-game description, that was given to Traveler by Zhongli as a symbolic gift, associated with him.",
            "reference" : """
                A Fortune Coin with a special Guyan Sigil engraved into it. It is said that this is the symbol of Rex Lapis, and can be used as a protective talisman.
            """
        },
        {
            "category" : "dialogue",
            "query" : "Create an in-world idle dialogue with Madame Ping. Player can have this dialogue with Madame Ping only after completing The Realm Within, so after obtaining Cleansing Bell, and it is like a small talk with a character. Write 5 blocks: each block consists of starting Traveler's (Player's) replica which is followed by 2-4 replicas by Madame Ping, each replica may contain from 1-2 sentences. Before Traveler's replicas write Traveler:, before Madame Ping's replicas write Madame Ping:. The final block must start with Goodbye from Traveler and character's answer to this.",
            "reference" : """
                Madame Ping: Oh? Why, hello child.
                Madame Ping: Though flowers bloom yet always fade, but people go yet return another day. I just knew that you'd be back to see me before long, hehe.
                
                Traveler: White Admiring the flowers?
                Madame Ping: That I am. One season after the next, this flower blooms, then falls, only to bloom again. And on it goes — just like Liyue.
                Madame Ping: Look over here, and over here. So many flowers are blooming! It looks like this will be a blessed year.
                
                Traveler: White What type of flowers are these?
                Madame Ping: These are Glaze Lilies. Probably the last surviving ones in all of Liyue Harbor.
                Madame Ping: People say that the sound of singing nourishes them, and helps them grow...
                Madame Ping: But the sound of fighting, of cursing... it is poison to them, and eats away at their roots.
                Madame Ping: That is why, over centuries of strife, they have disappeared from Liyue's landscape.

                Traveler: Tell me about the past.
                Madame Ping: Just because my complexion screams "ancient history" doesn't mean I love discussing the past, you know.
                Madame Ping: ...Relax, I'm only kidding.
                Madame Ping: Truth be told, the older you get, the more time you have on your hands, and the more you start contemplating the past... I do wonder how my old friends back in the mountains are getting on...
                Madame Ping: Liyue Harbor is really a wonderful place now. I wonder if they would ever want to step down into the mortal realm and get a taste of the lifestyle here...
                Madame Ping: Look at me rambling on again, I must sound like a lunatic! Maybe one day I can discuss my past with you... if it is meant to be.
                
                Traveler: This is a beautiful flower.
                Madame Ping: Hehe, all thanks to the beautiful weather we're having today.
                Madame Ping: I'd always hoped that one day this flower would bloom all over Liyue once more...
                Madame Ping: But in the meantime, would you take a few of these flowers from Yujing Terrace with you on your journey, so that they might see the world?

                Traveler: Goodbye.
                Madame Ping: Goodbye, young one. If you are ever passing by Jueyun Karst, please pass on my regards to my old friends.
                Madame Ping: If it is meant to be that you see them... then you will see them.
            """
        },
        {
            "category" : "dialogue",
            "query" : "Create an in-world idle dialogue with Chaonan, a man in Liyue Harbor. He gave up his dreams of joining the Crux Fleet to become a repairman. Start this dialogie with two replica's of Chaonan, then write Write 5 blocks: each block consists of starting Traveler's (Player's) replica which is followed by 2-4 replicas by Chaonan, each replica may contain from 1-2 sentences. Before Traveler's replicas write Traveler:, before Chaonan's replicas write Chaonan:. The final block must start with Goodbye from Traveler and character's answer to this. This dialogue is like a small talk with a character, revealing his personality and his dreams.",
            "reference" : """   
                Chaonan: ...Best if you say the bare minimum to me. I don't want to deplete my stamina.
                Chaonan: *sigh* If I'd known this would happen, I would have followed my dreams...

                Traveler: Are you tired?
                Chaonan: ...I'm a ship repairman. When the ships are at sea, I have nothing to do. But when they dock, I'm so busy I never get a wink of sleep...
                Chaonan: So I have decided to conserve my stamina from now on. Otherwise, I won't be able to cope when the busy period hits.
                
                Traveler: Have you considered a change of occupation?
                Chaonan: Well, this job is rather busy, but being busy also means that there's money to be made.
                Chaonan: We have Rex Lapis' divine predictions to thank for that. The number of merchants going out to sea has increased year on year in recent times, which has led to our business being on the rise, too.
                Chaonan: Those merchants who made their fortune aren't stingy, either. They often give us some tips on top of the repair fees.
                Chaonan: Hard work begets an equal reward, and for the lads that want to earn some money while they're young, this job is pretty good. We shouldn't waste the opportunity that Rex Lapis has given us.
                
                Traveler: What dreams?
                Chaonan: ...I once had the opportunity to join the Crux Fleet, but by the time I signed up, they were full.
                Chaonan: I gave up on being a seafarer in the end, and went into ship repair work instead. Now I can only watch as the ships go in and out.
                Chaonan: Maybe if I'd pushed a little harder for it back in the day, I wouldn't have ended up where I am now.
                Chaonan: And maybe the reason I've not been favored enough to receive a Vision is because I give up too easily.
                Chaonan: ...I hope you don't end up like me.
                
                Traveler: What a pity...
                Chaonan: The problem is my personality... That, and luck not being on my side.
                Chaonan: *sigh* It's funny, you know, the day before I signed up I even made a point of going out and buying this feather, which is supposed to bring good luck. Clearly, it had no effect whatsoever...
                Chaonan: ...Well, no use complaining about it. Here, you can have it, maybe it works better on adventurers.
                
                Traveler: Goodbye.
                Chaonan: Goodbye. Now to focus all my energy on getting a good rest...
            """
        },
        {
            "category" : "dialogue",
            "query" : "Create a dialogue for starting a quest about searching for relics in ruins. These relics contains fragments of Liyue's history and culture, and are located in Guili Assembly. The quest-giver is Soraya - a researcher from the Sumeru Akademiya who holds a great interest in Liyue's history and in particular, the legacy of Guizhong, the Lord of Dust. She is interested in relics and conflicts with Treasure Hoarders because they search relics to sell them. Write 5 blocks if replicas: each block starts with Travelers clarifying question and proceeds with 3-8 replicas of Soraya, that can contain 1-2 sentences. Before Traveler's replicas write Traveler:, before Soraya replicas write Soraya:.",
            "reference" : """
                Soraya: ...How many secrets does Liyue truly hold?
                
                Traveler: Secrets?
                Soraya: Oh, Traveler, it's you. The secrets I'm referring to are a set of inscriptions.
                
                Traveler: Do you mean buried treasure?
                Soraya: Haha, who knows? It's always possible. Talk of secrets waiting to be uncovered does tend to make one think of buried treasure.
                Soraya: This place we are in is called the Guili Assembly. According to Liyue folklore, it was once inhabited by people — in fact, it was rather prosperous.
                Soraya: But for various reasons, the whole place was wiped out. All that remains now is the countless ruins scattered across the landscape.
                Soraya: Some stories put it down to conflict between the gods, other people say it was a war between humans. And some sources talk of a sudden, catastrophic natural disaster...
                Soraya: Whatever the reason was, ruins are all that remains here now. The only ones who come here are the hilichurls, the Treasure Hoarders...
                Soraya: ...And masochistic scholars, like me, who in spite of all the risks just can't resist the idea of how much research value these ruins have to offer.
                Soraya: In practice though, scholars like me aren't cut out for this kind of field work! I've hardly covered any ground yet, and I already need a rest...
                Soraya: The actual ruin-exploring seems more your area of expertise. What do you say, fancy helping me out?
                
                Traveler: Sure.
                Soraya: Great. Hmm, but I should give you an idea of what to look for... Yes, keep your eyes peeled for stone pillars and stone tablets.
                Soraya: Given how ancient these ruins are, any text we can hope to find will be written on stone, not paper.
                Soraya: After all, stone is the oldest and most durable medium for written communication.
                
                Traveler: What am I looking for?
                Soraya: Hmm... Text essentially. Written messages from the past.
                Soraya: And if you're looking for text in ruins as ancient as these, that means looking for stone pillars and stone tablets.
                Soraya: Stone is the oldest, most durable medium for written communication. All the more so in the domain of the Geo Archon.
                Soraya: The stones here have borne witness to all of history. I am sure they will not disappoint.
                
                Traveler: Only if there's treasure...
                Soraya: Whatever treasure you find, you can keep. Material wealth isn't what appeals to me.
                Soraya: In Liyue there's a saying that goes: "Money belongs to the world outside your head." I'm more interested in acquiring knowledge than money.
                Soraya: So while you're hunting for treasure, please also keep your eyes peeled for stone pillars and stone tablets.
                Soraya: Given how ancient these ruins are, any knowledge we can hope to find will most likely be written on stone.
            """
        },
        {
            "category" : "dialogue",
            "query" : "Create an in-game short dialogue with Chef Mao, owner of the Wanmin Restaurant and father of Xiangling. This dialogue must contain three blocks. First is Chef Mao's start welcoming replica, that he says before player selects a dialogue option. Second block starts with Traveler's replica and proceeds with Chef Mao's answer, that contains 1-2 sentences; this block is about the restaurant itself. Third block is about Xiangling, her personality and talent that makes restaurant famous: this block starts with Traveler's question about Chef Mao's daughter and proceeds with 4-5 replicas of Chef Mao describing his Xiangling's talent in cuisine and her revolutionnary methods. Before Traveler's replicas write Traveler:, before Chef Mao replicas write Chef Mao:.",
            "reference" : """
                Chef Mao: Here for the Wanmin Restaurant experience? Hi, I'm Chef Mao. Everyone in Chihu Rock will have heard of me.

                Traveler: Tell me more about the Wanmin Restaurant.
                Chef Mao: Well, I'm the chef. I may not be quite the culinary legend that my daughter is, but you can be sure the Wanmin Restaurant is a cut above all those time-honored brands you see around!
                
                Traveler: Your daughter?
                Chef Mao: That's right — Xiangling! My dear daughter, and my best apprentice ever.
                Chef Mao: I only taught Xiangling how to cook so that she could inherit the Wanmin Restaurant from me one day. I had no idea she'd turn out to be such a pro. Her abilities far exceed my own.
                Chef Mao: But she's more than just an excellent chef. She has a real gift for innovation — she's constantly creating all these wonderful new dishes. That really sets her apart from what the time-honored brands do. And it makes her dishes all the more special for people who are used to traditional cuisine.
                Chef Mao: I started to worry that the restaurant might be too constricting for her. So I sent her off for a while to do some exploring and perfect her art.
                Chef Mao: If anyone is going to revolutionize Liyue cuisine in the future, it'll be her. No doubt about it!
            """
        },
        {
            "category" : "dialogue",
            "query" : "Create an in-game short dialogue with Sisi, a romantic girl who is waiting for her lover-sailor, with whom she wants to get engaged (but he appears to have lied to Sisi that he is devoted to her and will have to keep away from her due to the nature of his work). All she thinks about is love and romance. Write a start replica's of Sisi, then - three blocks of dialogues. Each block starts with Traveler's replica (question), and proceeds with 1-4 Sisi's replicas that may include 1-2 sentences. The first block is about Sisi's lover, the second block is about Rex Lapis' death and Sisi's thoughts about this, and the final block is goodbie-phrases of Traveler and then Sisi.Before Traveler's replicas write Traveler:, before Sisi' replicas write Sisi:.",
            "reference" : """
                Sisi: Oh hi. Look, sorry, if you're hitting on me I'll have to refuse. I'm waiting on someone.
                
                Traveler:  Who are you waiting for?
                Sisi: My lover, Chaoxi. He's a sailor.
                Sisi: We don't get to see each other very often, he's never back on land for very long...
                Sisi: He's already been gone for so long, who knows when he'll be back...
                Sisi: Next time he gets back I'm just going to ask him outright when we are going to get married. All this waiting is just too much...
                
                Traveler:  About Rex Lapis' death...
                Sisi: Rex Lapis and Liyue, like two lovers for thousands of years...
                Sisi: But all relationships must come to an end. If death is the thing that finally separates you, well then surely that's a sign that you had a good run...
                Sisi: ...or is it a sign that we have all been dumped by Rex Lapis?
                
                Traveler: Goodbye.
                Sisi: Farewell.
            """
        },
        {
            "category" : "location description",
            "query" : "Create a lore description for Qingxu Pool - a subarea located in Lisha, Liyue. Write unstructured text, firstly - 3-4 sentences about legend connected with Qingxu Pool's history connected with Archon War and Rex Lapis and adeptis, then - 1 sentence about its current state and 2-3 sentences about its modern history: how Qixing and other groups try to inspect ruins located there.",
            "reference" : """
                According to legend, the Qingxu Pool ruins were built before the Archon War. After the evil god ruling the city was defeated, the city was submerged underwater until the end of the Archon War. It is unclear whether the legend of this evil god are actually referring to the Geo Dragon Azhdaha, who rampaged across Lisha before Rex Lapis and three of the adepti finally sealed him in The Chasm. Records of that conflict have been memorialized in the three Nameless Treasures, one of which can be found in Qingxu Pool.
                While Qingxu Pool is no longer inhabited, a settlement which Tieshan hails from still exists around the area.
                In the past, the Liyue Qixing approved an attempt to excavate the Qingxu Pool ruins for their rumored buried treasure. However, the ruins were completely sealed shut and at the time, the technology needed to blast through solid rock did not yet exist, so the project was abandoned. 
            """
        },
        {
            "category" : "location description",
            "query" : "Write two short in-game descriptions about Dunyu Ruins, a subarea located in Lisha, Liyue. First description more lyrical, poetically desribing current appearance and past histories about ruins. This description contains 4-5 short sentences. Second description is an expedition report in 2 sentences: what is this place like and what possibly can be found there.",
            "reference" : """
                The Land of Hidden Jade. But the war of ages past is long over, and the mounds of glittering jade have long been destroyed. Now, only the whisper of flowing water remains.
                - Viewpoint, Dunyu Ruins

                Every corner of Dunyu Ruins is filled with ancient vestiges, suspended in time inside the dead lake.
                Without a doubt, there is still a large amount of treasure of unknown origin yet to be discovered.
                - Expedition description
            """
        },
        {
            "category" : "location description",
            "query" : "Create an environment element for Lisha, Liyue, connected with Yakshas. Describe it: what is it, where it is located, its appearance and also its changes: how it appears at first and how it changes with time and human efforts. As output, write only a short and simple descriprion of 3-4 sentences.",
            "reference" : """
                Pervases' Temple is a point of interest located in Lisha, Liyue.
                It is a temple dedicated to the deceased Yaksha, Pervases, and contains a statue dedicated to him. It originally appears as a dilapidated house, before being restored.
            """
        },
        {
            "category" : "location description",
            "query" : "Create a location that is a part of a Liyue: The Chasm, mining region that provided essential ores but has a great history related to adeptis and Zhongli, and also more ancient civilizations. Write a short descrsiption for this newly created location, describe its appearance and elements in 3-4 sentences, then describe its modern uses or problems its facing now (also 3-4 sentences), and finally write its history, legends and describe its participation in events of Liyue history (5-7 sentences).",
            "reference" : """
                The Chasm is a massive, Liyue-based mining region, characterized by its deep, open-pit surface, contrasting with a dark, sprawling underground filled with glowing flora, ancient ruins, and hazardous black mud. It is known for its intense orange-brown dirt, steep rocky terrain, and a mysterious upside-down city.
                In the recent past, a series of mysterious accidents led the entire mine to be closed, putting its workers out of a job. However, the Liyue Qixing is gradually reopening the area, thus allowing miners to resume working. Beneath the Seven-Star Array is the Underground Mines.
                According to legend, the Chasm was originally was the ancient civilization of Lang-Gan. The civilization was destroyed and the Chasm would be formed by the Solar Chariot during the War of Funerary Flame. This meteorite had a "proud and agitated temper," and during the Archon War, the constant strife caused this meteorite to leap back into the heavens, leaving behind the massive expanse that was the Chasm. Legends also said that this was the area where Morax and Azhdaha clashed in battle. Most recently, during the cataclysm 500 years ago, an "iron meteor" fell into The Chasm, which marked the beginning of the Abyss' invasion.
            """
        },
        {
            "category" : "location description",
            "query" : "Describe a new location with name Chenyu Vale located on the north of Liyue, related to tea production and creation of jade figurines. This region has his own adeptis. Write a short description for this location: 1-2 sentences of appearance and landscapes, 2-3 sentences of history and culture, and 2-3 key locations important for this region.",
            "reference" : """
                Chenyu Vale is a lush, northwestern region of Liyue, characterized by mountainous tea-growing landscapes, tea-house culture, and flowing rivers. 
                It was protected by the adeptus Fujin and the suanni Lingyuan under an unnamed goddess who turned maddened by the Archon War. Following the creation of The Chasm 6000 years ago, surviving tribes moved to this area, where Fujin taught them to work with jade and agriculture.
                Key Locations:
                 - Qiaoying Village: Famous for its tea.
                 - Yilong Wharf: A busy port and shipping hub for the tea trade.
                 - Mt. Yaojun: A central, sacred mountain.
            """
        },
        {
            "category" : "quest",
            "query" : "Create a world quest about Guili Plains, Guili Assembly ruins and Guizhong's history. The quest-giver is Soraya - a researcher from the Sumeru Akademiya who holds a great interest in Liyue's history and in particular, the legacy of Guizhong, the Lord of Dust. Quest must be related to her researches, in which Traveler helps her. The quest must contain 5 stages-actions: start (where to find Soraya, very short unstructured description of main points of dialogue a what starts the quest), actions - three staged in which player (Traveler) have to do something. If it is action related to some goals to achieve the quest, write what has to be done. If it is another dialogue, write short summary (4-5 sentences maximum) of the dialogue. Final stage is another dialogue that finishes the quest after completing all goals: write summary of this dialogue (3-4 sentences). Structure your output as a numbered list of stages: dialogues and player's tasks.",
            "reference" : """
                Start the quest by talking to Soraya near the ruin in Guili Plains. Soraya asks the Traveler to help her explore the Guili Assembly ruins and search for ancient inscriptions on stone pillars and tablets that could reveal Liyue's history. She is primarily interested in acquiring knowledge rather than material wealth, so she allows the Traveler to keep any treasure they might find during their search.         
                1. Head to Guili Assembly to search for an ancient stone tablet (0/5). Obtain Transcription Transcription from a stone tablet ×5
                2. Talk to Soraya. After the Traveler brings Soraya five stone tablets, she pieces together their fragmented text to reveal that the Guili Assembly was jointly protected by two allied gods—Guizhong, who ruled over Dust, and a Geo Archon—and that its name derives from syllables of their divine names. Fascinated by this account of peaceful cooperation between deities, Soraya wonders what catastrophic event could have destroyed the settlement despite their power and asks the Traveler to continue searching for more tablets to uncover the rest of the story.
                3. Search for the stone tablets (0/2). Obtain Transcription Transcription from a ruin ×2
                4. Talk to Soraya. The newly discovered tablets reveal that a massive battle destroyed the Guili Assembly and hint at hidden treasure protected by Guizhong's Four Commandments. Unable to immediately decipher the philosophical riddle, Soraya decides to return to Wangshu Inn with the Traveler to rest before continuing their investigation.
            """
        },
        {
            "category" : "quest",
            "query" : "Create a world quest that can be taken in Wangshu Inn. The quest may be about some sort of business I think, and connected with one character Landa who is a merchant. Quest consisits of 4 parts-stages: first one is start of the quest, describe in 2-3 senteces where and how Traveler takes this quest, if there is a dialogue, write a short summary - main points - of the dialoque. Then there must be three parts: stages 2 and 3 are player's goals or dialogues, describe strictly the goals or create short description for dialogues and their main points. Final stage (4th) is a dialogue with Landa after completing all the goals and taksks: write short summary for this dialogue (3-4 sentences). Be very simple, do not overcreate this quest and add some difficult dramatice legend to it: it is just a simple world side quest to make this world more live. Structure quest stages as numbered list.",
            "reference" : """
                Talk to Landa at Wangshu Inn to start the quest. Landa, a merchant who frequently travels between Mondstadt and Liyue, begs the Traveler for help after being ambushed by monsters and forced to flee, leaving behind a box containing crucial wine trade invoices. He anxiously asks the Traveler to retrieve the documents from the marked locations along his escape route, fearing he will lose his job if he fails to deliver them to his wealthy Liyue employer.
                1. Get the invoices (0/3)
                2. Talk to Bao'er. One Box of Goods Invoices is held by Bao'er, who will trade it in exchange for three Noctilucous Jade. Bao'er admits to finding the lost box of invoices but proposes a barter: she will only return the documents in exchange for 3 Noctilucous Jades as compensation for her time guarding them. After the Traveler agrees to the deal and provides the gems, she hands over the invoices, advising them not to ask too many questions about the high-value transactions listed inside before sending them on their way.
                3. Talk to Landa. Landa gratefully accepts the recovered invoices from the Traveler and provides their reward, but panics upon noticing the box has been opened and learning that a stranger examined its contents. He urgently excuses himself to contact his boss about the potential security breach, assuring the Traveler that he trusts them despite the confidential nature of the documents.
            """
        },
        {
            "category" : "quest",
            "query" : "Create a simple one-staged quest for Wuwang Hill, Liyue. In this quest preserve dark and mystic atmoshpere of a place, but remains very simple do not overcreate some legends. Output only short description of a quest: how to take it, what about is the first dialogue with quest-giver if any; then - player's goal (task), what he or she has to do to accomplish the quest, and finally - what is the end of the quest, how it finishes. Add in the finish some replicas of Paimon that she will say to finish the quest (start them with Paimon:). Write no more than 8-10 sentences at all.",
            "reference" : """
                Book in the Woods is a World Quest in Wuwang Hill, Liyue.
                Talk to Little Nine. Little Nine asks the Traveler to help find a precious manuscript written by their neighbor Big Nine—a book they were allowed to preview before publication but accidentally lost after falling into a river and drying it in the woods. Worried that Big Nine will be heartbroken or angry if the book is gone for good, Little Nine directs the Traveler to search the nearby forest where they last remember leaving it to dry.
                1. Find the lost book in the forest (0/2). Traveler and Paimon cannot find any books and decided to go back to ask for extra clues.
                2. Go back and find Little Nine. Little Nine will not be there and the quest will finish.
                Finish dialogue:
                Paimon: What the... Where'd that girl go?
                Paimon: Has she gone to find the book herself?
            """
        },
        {
            "category" : "quest",
            "query" : "Create a quest located in Mingyun Village, Liyue, that reveals some story related to the villagers that abandoned the place long time ago. This story should be very simple, may be connected with some family conflicts or something like that. Write 5 stages of the quest as a numbered list: first stage is starting point, describe how to start the quest. Then stages may be player's task and actions, or dialogues: for dialogues, write short description of what dialogue is about (1-3 sentences). In the final stage describe also a reward that Traveler and Paimon get. Write no more than 15-17 sentences at all.",
            "reference" : """
                1. Start the quest by talking to Yuan Hong in Mingyun Village. Yuan Hong, lost in melancholic reverie, recites cryptic poetry about autumn leaves and solitude while dismissing worldly wealth as a farce. He expresses a deep longing to leave the village and start life anew, barely acknowledging the Traveler's attempts to engage him in conversation.
                2. Look around Mingyun Village. Go to the house facing the billboard and read the "Lost Notes". It is required to read the notes as the first step, as it triggers the spawning of the four parts of the fathers will and the matching four markers on the map.
                3. Look for a will (0/4). During this, you can find three people in different locations (these are three brothers, sons of Yuan Hong, now all three of thems searching the treasure without desire to reunir as a family) and listen to them:
                    - Yuan Cheng. Yuan Cheng vents his frustration over his older brothers hoarding the best discoveries and his father's willful blindness to the family's dynamics. Convinced that a hidden treasure lies in the mine unbeknownst to them, he resolves to claim it for himself and flee far away from his family.
                    - Yuan Qing. Yuan Qing laments his misfortune after sneaking away from his family to be the first to reach the mine and claim its hidden treasure, only to be ambushed by monsters. Now trapped and overwhelmed, he fear he is doomed to fail.
                    - Yuan Liang. Yuan Liang mocks his brothers for misinterpreting their father's will and arrogantly dismisses any sense of familial duty, focusing solely on the line mentioning hidden treasure. Consumed by greed, he plots to claim the mine's riches for himself, confident that his siblings' foolishness will leave the prize entirely to him.
                4. Find the treasure. The treasure is at the base of a large tree. When you approach the tree, four Treasure Hoarders will spawn.
                5. Collect the treasure. A dig option becomes available at the base of the tree. After digging, a Precious Chest will spawn.
            """
        },
        {
            "category" : "quest",
            "query" : "Create a world side-quest about restoring the abandoned temple of Pervases, ancient Yaksha and a friend of Xiao. This quest is related to Wang Ping'an, previous scammer who decided to change his life for good. Output unstructured text with quest progress description: describe what Traveler and Paimon do, where they go, who they meet, with whom they have conversations, etc. Include three parts of temple reconstruction: starting the reconstruction, proceeding with it, and, finally, finishing works. On each step Traveler have to help Wang Ping'an with his works somehow, so describe what should be done on each stage. Write about 20-30 sentences in total.",
            "reference" : """
                As the Traveler and Paimon walk by Wanmin Restaurant, they smell the aroma of Grilled Tiger Fish and decide to buy some. The Traveler buys three servings and Paimon wonders who's the third serving for. The Traveler had thought of Pervases and decided to buy another serving and go pay their respects. Paimon hopes the statue of Pervases hasn't been defaced as the temple was in a pretty sad state with Treasure Hoarders nearby the last time they were there.
                As they approach the temple, they see Xiao nearby. Paimon wonders out loud why no one maintains the temple even though the Yakshas are heroes who defended Liyue Harbor. Xiao goes silent and Paimon wonders if he's angry but he says that they did not perform their duty for recognition. They then hear someone calling for help so Xiao and the Traveler go to assist. After defeating the enemies, they find the person under attack was Wang Ping'an, previously known to them as "Starsnatcher." Wang Ping'an had turned a new leaf and was on his way to restore Pervases's temple. Xiao leaves and the Traveler and Wang Ping'an head to the temple to continue their discussion about the temple. Paimon and Traveler wonder if Wang Ping'an was actually being genuine and so Wang Ping'an tells them to come back after work has started to see he has indeed changed.
                The Traveler returns to the temple later and find that Wang Ping'an has brought some workers and are making good progress on the temple. The Ministry of Civil Affairs got wind of what was going on and posted some Millelith to guard the temple throughout the construction. Although the Millelith are guarding the construction site, Wang Ping'an is worried that material deliveries might get attacked by nearby Treasure Hoarders and asks the Traveler to handle them. The Traveler swiftly defeats the Treasure Hoarders and they ask what they did to offend the Traveler. So the Traveler informs them they are renovating a Yaksha's temple. The Treasure Hoarders did not know the dilapidated temple was for an Adeptus and promise not to mess with the temple lest the Adepti judge them. After reporting back to Wang Ping'an, the Traveler is thanked and is asked to gather some Sandbearer Wood next. Wang Ping'an tells the Traveler that construction will take some more time and to come back in some time.
                When returning to the temple again, the Traveler finds the construction has finished and the temple fully repaired. Wang Ping'an asks the Traveler if they'd like to offer some incense and so the Traveler lights some incense at the censer. Wang Ping'an tells the Traveler that he will remain at the temple as he compiles his book about the Vigilant Yaksha and welcomes the Traveler to drop by anytime to offer incense.
            """
        }
    ]
}

TEST_DATA_RUSSIAN_SCENARIO = {}