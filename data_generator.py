import random


def get_topic_words():
    return {
        "tech": [
            "software", "programming", "AI", "machine learning", "cloud", "data",
            "computer", "algorithm", "database", "API", "internet", "network",
            "security", "encryption", "mobile", "app", "web", "server", "code"
        ],
        "sports": [
            "football", "basketball", "soccer", "tennis", "swimming", "running",
            "olympics", "championship", "team", "player", "score", "goal",
            "match", "game", "coach", "training", "fitness", "exercise"
        ],
        "science": [
            "biology", "chemistry", "physics", "astronomy", "geology", "research",
            "experiment", "scientist", "hypothesis", "theory", "lab", "discovery",
            "space", "planet", "earth", "nature", "evolution", "dna"
        ],
        "health": [
            "doctor", "hospital", "medicine", "health", "wellness", "exercise",
            "diet", "nutrition", "disease", "treatment", "vaccine", "virus",
            "mental health", "therapy", "fitness", "gym", "yoga", "sleep"
        ],
        "business": [
            "company", "business", "market", "economy", "finance", "investment",
            "startup", "entrepreneur", "product", "service", "customer", "sales",
            "marketing", "management", "leadership", "strategy", "growth"
        ],
        "entertainment": [
            "movie", "film", "music", "song", "artist", "book", "novel", "game",
            "video game", "TV", "show", "celebrity", "theater", "dance", "art"
        ],
        "travel": [
            "travel", "trip", "journey", "destination", "hotel", "flight",
            "vacation", "tourism", "country", "city", "beach", "mountain",
            "adventure", "explore", "culture", "food", "cuisine"
        ],
        "education": [
            "school", "college", "university", "student", "teacher", "learning",
            "education", "course", "degree", "class", "study", "knowledge",
            "skill", "training", "online", "classroom", "homework"
        ]
    }


def get_sentence_templates():
    return [
        "Many people are interested in learning about {topic_word}",
        "{topic_word} has become increasingly popular in recent years",
        "Understanding {topic_word} can be very beneficial",
        "There are many aspects to {topic_word} that people enjoy",
        "New developments in {topic_word} are happening all the time",
        "Experts often discuss the future of {topic_word}",
        "Learning about {topic_word} can open new opportunities",
        "{topic_word} plays an important role in our daily lives",
        "People have different opinions about {topic_word}",
        "The history of {topic_word} is fascinating to explore",
        "Modern technology has transformed how we approach {topic_word}",
        "{topic_word} is a topic that many are passionate about",
        "Practicing {topic_word} can improve your skills significantly",
        "The best way to learn {topic_word} is through experience",
        "Community plays an important role in {topic_word}",
        "There are many resources available for {topic_word}",
        "{topic_word} continues to evolve and change over time",
        "Different cultures approach {topic_word} in unique ways",
        "The benefits of {topic_word} are well-documented",
        "{topic_word} requires both knowledge and practice"
    ]


def generate_large_corpus(min_samples=1500):
    topics = get_topic_words()
    templates = get_sentence_templates()
    corpus = []
    
    for _ in range(min_samples):
        # Pick a random topic
        topic_category = random.choice(list(topics.keys()))
        topic_words = topics[topic_category]
        
        # Pick 2 random topic words to make it more diverse
        word1 = random.choice(topic_words)
        word2 = random.choice(topic_words)
        
        # Pick a random template
        template = random.choice(templates)
        
        # Fill in the template with a word (randomly choose between word1 and word2)
        sentence = template.format(topic_word=random.choice([word1, word2]))
        
        corpus.append(sentence)
    
    return corpus


if __name__ == "__main__":
    print("Generating large corpus...")
    corpus = generate_large_corpus(min_samples=5000)
    print(f"Generated {len(corpus)} samples")
    with open("data/large_corpus.txt", "w", encoding="utf-8") as f:
        for text in corpus:
            f.write(text + "\n")
    print("Corpus saved to data/large_corpus.txt")
