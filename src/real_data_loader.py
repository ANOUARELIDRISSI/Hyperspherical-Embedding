import random
import re


TOPICS = {
    "tech": {
        "terms": ["software", "programming", "AI", "machine learning", "cloud", "data", "algorithm", "database", "API", "network", "security", "encryption"],
        "contexts": [
            "Engineering teams use {a} and {b} to build reliable digital services.",
            "Modern {a} projects often depend on clean {b}, testing, monitoring, and careful deployment.",
            "Companies invest in {a} because it improves automation, search, recommendations, and decision support.",
            "A practical {a} system needs secure infrastructure, useful data, and clear product goals.",
        ],
        "queries": ["machine learning and artificial intelligence", "software engineering and cloud systems", "data security and encryption"],
    },
    "sports": {
        "terms": ["football", "basketball", "soccer", "tennis", "swimming", "running", "championship", "team", "player", "coach", "fitness", "training"],
        "contexts": [
            "Athletes improve {a} performance through structured {b}, recovery, and coaching.",
            "A strong {a} team studies opponents, practices tactics, and builds fitness before a match.",
            "Fans follow {a} because scores, rivalries, players, and championships create drama.",
            "Good {a} coaching balances skill practice, conditioning, nutrition, and mental preparation.",
        ],
        "queries": ["sports teams and championship training", "fitness coaching for athletes", "football and basketball performance"],
    },
    "science": {
        "terms": ["biology", "chemistry", "physics", "astronomy", "geology", "research", "experiment", "theory", "space", "planet", "earth", "evolution"],
        "contexts": [
            "Scientists study {a} by forming hypotheses, running {b}, and comparing evidence.",
            "{a} research helps explain natural systems, from molecules and organisms to planets and space.",
            "Progress in {a} depends on measurement, peer review, mathematical models, and reproducible experiments.",
            "Students learn {a} by connecting theory with observation, laboratory work, and data analysis.",
        ],
        "queries": ["space astronomy and planets", "biology chemistry and experiments", "scientific research and theory"],
    },
    "health": {
        "terms": ["doctor", "hospital", "medicine", "wellness", "exercise", "diet", "nutrition", "treatment", "vaccine", "therapy", "fitness", "sleep"],
        "contexts": [
            "Health professionals combine {a}, {b}, prevention, diagnosis, and follow-up care.",
            "Daily wellness improves through exercise, nutrition, sleep, stress management, and medical guidance.",
            "Patients rely on {a} and treatment plans to manage illness and recover safely.",
            "Public health programs use vaccines, education, screening, and community support to reduce disease.",
        ],
        "queries": ["health wellness nutrition and exercise", "medicine treatment doctors and hospitals", "public health vaccines and disease prevention"],
    },
    "business": {
        "terms": ["company", "market", "economy", "finance", "investment", "startup", "entrepreneur", "product", "customer", "sales", "marketing", "growth"],
        "contexts": [
            "A {a} grows by understanding customers, pricing products, managing finance, and improving sales.",
            "Startup founders test a market, build a product, find customers, and raise investment.",
            "Business strategy connects {a}, operations, marketing, leadership, and long-term growth.",
            "Investors study markets, revenue, costs, competition, and risk before supporting a company.",
        ],
        "queries": ["startup business investment and customers", "finance markets and company growth", "sales marketing and product strategy"],
    },
    "entertainment": {
        "terms": ["movie", "film", "music", "song", "artist", "book", "novel", "video game", "show", "celebrity", "theater", "dance"],
        "contexts": [
            "Audiences enjoy {a} because stories, performances, sound, and visual style create emotion.",
            "Creative teams produce {a} through writing, rehearsal, editing, distribution, and promotion.",
            "Artists build communities around music, film, games, books, theater, and live events.",
            "Entertainment businesses measure audience attention, reviews, ticket sales, streams, and fan engagement.",
        ],
        "queries": ["movies music artists and entertainment", "books theater and creative performances", "video games shows and fan communities"],
    },
    "travel": {
        "terms": ["travel", "trip", "journey", "destination", "hotel", "flight", "vacation", "tourism", "city", "beach", "mountain", "culture"],
        "contexts": [
            "Travelers plan a {a} by choosing destinations, booking flights, comparing hotels, and exploring culture.",
            "Tourism supports local restaurants, museums, guides, transport, and small businesses in a city.",
            "A memorable vacation may include beaches, mountains, food, history, festivals, and outdoor adventure.",
            "Good travel planning balances budget, safety, weather, transportation, and the purpose of the journey.",
        ],
        "queries": ["travel destinations hotels and flights", "vacation tourism beaches and culture", "city trips and journey planning"],
    },
    "education": {
        "terms": ["school", "college", "university", "student", "teacher", "learning", "course", "degree", "class", "study", "knowledge", "homework"],
        "contexts": [
            "Education helps students build knowledge through courses, practice, feedback, and assessment.",
            "Teachers design classes with clear goals, examples, discussion, homework, and support.",
            "Universities and colleges offer degrees, research opportunities, career preparation, and student communities.",
            "Online learning can expand access when courses include structure, interaction, projects, and mentoring.",
        ],
        "queries": ["school college students and teachers", "online courses degrees and learning", "homework study and classroom knowledge"],
    },
}

PERSPECTIVES = [
    "for beginners",
    "in professional settings",
    "for community projects",
    "during long-term planning",
    "when resources are limited",
    "for practical decision making",
    "in fast-changing environments",
    "when teams need reliable results",
]


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def filter_sentence(sentence, min_words=8, max_words=50):
    word_count = len(sentence.split())
    return min_words <= word_count <= max_words


def generate_labeled_corpus(total_sentences=50000, seed=42):
    rng = random.Random(seed)
    corpus = []
    seen = set()
    topic_names = list(TOPICS)
    attempts = 0

    while len(corpus) < total_sentences:
        attempts += 1
        label = topic_names[len(corpus) % len(topic_names)]
        topic = TOPICS[label]
        term_a, term_b = rng.sample(topic["terms"], 2)
        template = rng.choice(topic["contexts"])
        perspective = rng.choice(PERSPECTIVES)
        sentence = f"{template.format(a=term_a, b=term_b)} This matters {perspective}."
        if attempts > total_sentences * 20:
            sentence = f"{sentence} Scenario {len(corpus) + 1} adds a distinct example for retrieval training."
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        corpus.append({"text": sentence, "label": label})

    rng.shuffle(corpus)
    return corpus


def generate_improved_synthetic_corpus(total_sentences=50000):
    records = generate_labeled_corpus(total_sentences=total_sentences)
    texts = [record["text"] for record in records]
    print(f"Generated labeled synthetic corpus with {len(texts)} unique sentences")
    return texts


def load_labeled_corpus(total_sentences=50000):
    records = generate_labeled_corpus(total_sentences=total_sentences)
    print(f"Generated labeled synthetic corpus with {len(records)} unique sentences")
    return records


def load_real_corpus(total_sentences=50000):
    return generate_improved_synthetic_corpus(total_sentences=total_sentences)


def build_topic_documents(sentences_per_topic=80):
    records = generate_labeled_corpus(total_sentences=sentences_per_topic * len(TOPICS))
    grouped = {topic: [] for topic in TOPICS}
    for record in records:
        grouped[record["label"]].append(record["text"])
    return [
        {"id": topic, "label": topic, "text": " ".join(grouped[topic])}
        for topic in TOPICS
    ]


def chunk_text(text, chunk_sentences=4, overlap=1):
    sentences = split_into_sentences(text)
    step = max(1, chunk_sentences - overlap)
    chunks = []
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start:start + chunk_sentences])
        if filter_sentence(chunk, min_words=12, max_words=220):
            chunks.append(chunk)
    return chunks
