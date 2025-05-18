import pandas as pd, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── load & clean ─────────────────────────────────────────────────────
food  = pd.read_csv("food.csv")
rates = pd.read_csv("ratings.csv")

food["C_Type"] = food["C_Type"].str.strip().str.title()
food["Veg_Non"] = food["Veg_Non"].str.lower()

rating_stats = (rates.groupby("Food_ID")["Rating"]
                      .agg(mean="mean", n="size")
                      .reset_index())
food = food.merge(rating_stats, on="Food_ID", how="left").fillna({"mean": 0, "n": 0})

# ── TF‑IDF vectors of descriptions ───────────────────────────────────
vec   = TfidfVectorizer(stop_words="english")
tfidf = vec.fit_transform(food["Describe"])

# ── known cuisine/category words ─────────────────────────────────────
known_categories = [
    "indian", "healthy food", "dessert", "chinese", "italian", "snack",
    "thai", "french", "mexican", "japanese", "beverage", "nepalese",
    "korean", "vietnames", "spanish"
]

# ── Eliza‑style regex patterns ───────────────────────────────────────
patterns = [
    (r"\bnon[- ]?(veg|vegetarian)\b", lambda m, s: s.update(veg="non-veg")),
    (r"\bveg(etarian)?\b",            lambda m, s: s.update(veg="veg")),
    (r"\bvegan\b",                    lambda m, s: s.update(veg="veg")),
    (r"\bi (dislike|hate) (\w+)",     lambda m, s: s["ban"].add(m.group(2))),
    (r".*",                           lambda m, s: None),
]

# ── conversation state ───────────────────────────────────────────────
def new_state():
    return {"veg": None, "ct": set(), "ban": set(), "keywords": set(), "anything": False}

# ── scoring function ─────────────────────────────────────────────────
def score(state, sim_threshold=0.10):
    mask = pd.Series(True, index=food.index)

    if state["veg"]:
        mask &= food["Veg_Non"].eq(state["veg"])

    if state["ct"]:
        # Check if any of the cuisines match
        ct_pattern = "|".join(map(re.escape, state["ct"]))
        mask &= food["C_Type"].str.contains(ct_pattern, case=False, na=False)

    cand = food[mask].copy()
    if cand.empty:
        return []

    # Similarity bonus
    if state["keywords"]:
        q_vec = vec.transform([" ".join(state["keywords"])])
        sims = cosine_similarity(q_vec, tfidf[cand.index]).ravel()
        cand["sim"] = sims
        if not state["anything"] and sims.max() < sim_threshold:
            return []
    else:
        cand["sim"] = 0.0

    # Ban filter
    if state["ban"]:
        ban_pat = "|".join(map(re.escape, state["ban"]))
        cand = cand[~cand["Describe"].str.contains(ban_pat, case=False, na=False)]
        if cand.empty:
            return []

    cand["final"] = cand["mean"] + cand["sim"] + 0.1 * cand["n"]
    return (cand.sort_values("final", ascending=False)
                [["Name", "C_Type", "Veg_Non", "mean"]]
                .head(3)
                .to_dict("records"))

# ── tiny stopword list for keyword capture ───────────────────────────
stop = {"i", "want", "something", "give", "food", "like", "need", "crave", "have", 
        "with", "that", "please", "and", "or", "maybe", "just"}

# ── main loop ────────────────────────────────────────────────────────
state = new_state()
print("Hi! Tell me what you feel like eating today. (type 'exit' to quit)")
while True:
    user = input("You: ").strip().lower()
    if user in {"exit", "quit"}:
        print("Bot: Okay, bye! 😊")
        break

    # Detect cuisines/categories
    for word in known_categories:
        if re.search(rf"\b{word}\b", user):
            state["ct"].add(word.title())

    if "anything" in user:
        state["anything"] = True

    # Run pattern matchers
    for pat, fn in patterns:
        m = re.search(pat, user, re.I)
        if m:
            fn(m, state)
            break

    # Extract keywords (≥4 letters, not stopwords)
    words = {w for w in re.findall(r"\b[a-z]{4,}\b", user) if w not in stop}
    state["keywords"].update(words)

    # Debug state
    print(f"Debug: Current state: {state}")

    recs = score(state, sim_threshold=0.05 if state["anything"] else 0.10)
    if recs:
        print("Bot: How about trying:")
        for r in recs:
            print(f"  • {r['Name']} ({r['C_Type']}, {r['Veg_Non']})")
        state = {"veg": None, "ct": set(), "ban": state["ban"], "keywords": set(), "anything": False}
    else:
        print("Bot: Tell me more about what you’d like…")
