TEST_TASKS = [
    {
        "id": "test",
        "query": "Write a short story for a future quest (do not structure it now) about Morgiana and her ordinary life.",
        "world_context": """
                Ali Baba, a poor woodcutter living in Persia, accidentally discovers the secret cave of forty thieves that opens with the magic words “Open, Sesame.” 
                He takes some of their treasure, but after sharing the secret with his greedy brother Cassim, tragedy follows: Cassim forgets the magic words, is trapped in the cave, and killed by the robbers. 
                With the help of the clever slave Morgiana, Ali Baba secretly buries his brother. 
                The captain of the thieves tries to kill Ali Baba by hiding his men in oil jars, but Morgiana outwits him by pouring boiling oil into the jars. 
                When the captain later disguises himself again, Morgiana recognizes him and kills him during a feast. 
                Thanks to her loyalty and intelligence, Ali Baba survives, gains great wealth, and rewards Morgiana with marriage to his son.
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