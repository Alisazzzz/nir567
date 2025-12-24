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