# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works
Explain your design in plain language.
Real-world recommenders like Spotify combine two strategies: collaborative filtering (learning from what millions of similar users played) and content-based filtering (matching songs by their audio attributes). This  focuses entirely on content-based filtering: no play history, no user comparisons, just a direct match between a song's measurable traits and a user's stated preferences.

Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
  - genre, mood, energy, acousticness, valence, tempo_bpm,  danceability, popularity, release_decade, detailed_mood, instrumentalness, livenesss
- What information does your `UserProfile` store
  - favorite genre, favorite mood, energy and likes acoustic
- How does your `Recommender` compute a score for each song
  - score = genre_match × 3.0
      + mood_match  × 2.5
      + (1 - |song.energy - user.target_energy|²)       × 2.0
      + (1 - |song.acousticness - user.target_acoustic|²) × 1.5
      + (1 - |song.valence - 0.70|²)                    × 0.5
      Features such as genre and mood are an exact match. Other features use a squared proximity formula so scores of 1.0 for a perfect match, curving steeply down as distance grows. Maximum possible score is 9.5.
      One bias that I realized from what the ai gave me was that the valence term uses a hardcoded target of 0.7, meaning the system always steers toward moderately positive songs regardless of what the user actually wants. A user who prefers dark or melancholic music will be silently penalized, since they have no way to set their own valence preference.
- How do you choose which songs to recommend
 - Every song in the catalog/ song file would be scored, then sorted, and the top k songs with their scores and explanations will be recommended.


You can include a simple diagram or bullet list if helpful.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"


Pictures:

### Baseline — pop / happy / energy 0.8 (original scorer, max 9.5)
![Baseline pop/happy/0.8 terminal output showing top 5 recommendations scored out of 9.5](pictures/Screenshot%202026-04-15%20at%201.28.59%20AM.png)

### Baseline 2 — pop / happy / 0.85 and Baseline 3 — lofi / chill / acoustic
![VS Code terminal showing baseline 2 pop/happy results and start of lofi/chill profile](pictures/Screenshot%202026-04-15%20at%201.41.11%20AM.png)

### Baseline 3 — lofi / chill / acoustic and Baseline 4 — rock / intense
![Terminal output for lofi chill acoustic profile and rock intense profile](pictures/Screenshot%202026-04-15%20at%201.41.18%20AM.png)

### Baseline 4 — rock / intense and ADV 1 — mood 'sad' not in catalog
![Rock intense results and start of adversarial profile with missing mood label](pictures/Screenshot%202026-04-15%20at%201.41.25%20AM.png)

### ADV 1 — missing mood and ADV 2 — high energy lofi conflict
![ADV 1 sad mood scoring zero on mood dimension and ADV 2 lofi genre with energy 0.9](pictures/Screenshot%202026-04-15%20at%201.41.30%20AM.png)

### ADV 2 — lofi energy conflict and ADV 3 — out-of-range energy 1.5
![ADV 2 showing lofi filter bubble and ADV 3 showing negative energy scores from out-of-range input](pictures/Screenshot%202026-04-15%20at%201.41.34%20AM.png)

### ADV 3 — out-of-range energy and ADV 4 — wrong capitalisation
![ADV 3 energy 1.5 results and ADV 4 showing Pop/Happy capitalisation breaking genre and mood match](pictures/Screenshot%202026-04-15%20at%201.41.38%20AM.png)

### ADV 4 — capitalisation fix and ADV 5 — genre 'metal' absent from catalog
![ADV 4 after case-insensitive fix and ADV 5 showing metal genre with hardcoded valence bias](pictures/Screenshot%202026-04-15%20at%201.41.41%20AM.png)

### ADV 5 — metal/angry results and ADV 6 — empty profile
![ADV 5 metal angry profile and start of ADV 6 empty profile ranked only by hardcoded valence](pictures/Screenshot%202026-04-15%20at%201.41.45%20AM.png)

### ADV 6 — empty profile (valence-only ranking)
![ADV 6 empty profile showing all songs scored purely by proximity to hardcoded valence target 0.70](pictures/Screenshot%202026-04-15%20at%201.41.48%20AM.png)
