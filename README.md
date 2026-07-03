# Spotify Audio Analysis Pipeline

This repository contains a comprehensive Exploratory Data Analysis (EDA) of the Spotify Songs dataset (TidyTuesday 2020). This project was developed as part of the **GDGOC AI Module 1** hands-on task, focusing on data cleaning, statistical analysis, and manual implementation of Machine Learning concepts using **NumPy** and **Pandas**.

## Project Overview
The primary goal was to move beyond simple function calls and understand the mathematical foundations of audio analysis. This pipeline explores over 30,000 tracks to identify trends in popularity, genre behavior, and audio feature correlations.

### Key Technical Highlights:
*   **Manual NumPy Operations:** Implementation of IQR and Min-Max scaling without external ML libraries like `scikit-learn`.
*   **AI-Assisted Workflow:** Integration of Gemini/GitHub Copilot for professional documentation and code refactoring.
*   **Advanced Vectorization:** Use of NumPy broadcasting for pairwise Euclidean distance matrices between genres.
*   **Production-Ready Code:** Refactoring the entire analysis into a modular `SpotifyAnalyzer` Python class.

---

## The Implementation Journey (Learning Log)

*   **stage 1: data integrity matters**
    > ive just learned that identifying the right column for deduplication is critical. at first, i considered using track_name, but then i realized different songs can have the same name. using track_id proved much more reliable for ensuring a unique dataset. i successfully removed 4,477 duplicate entries.
*   **stage 2: the pop paradox**
    > during genre analysis, i found that the **pop** genre has the highest popularity variance (std dev: 24.62). this taught me that pop is a "winner-takes-all" genre; streaming services cant just recommend any random pop song because the quality varies so much.
*   **stage 3: manual scaling and correlations**
    > implementing manual min-max scaling across nine audio features was a great exercise in vectorized math. i discovered that **energy and loudness** have a strong positive correlation (0.68), while **energy and acousticness** are strongly negative (-0.55).
*   **stage 4: human-ai collaboration**
    > using gemini to generate docstrings was a huge time-saver. however, i learned to stay critical—i had to catch a `TypeError` in the initial ai-suggested logic for `np.eye()` and fix it by using the proper shape tuple.

---

## Key Insights from the Data
Based on the analysis of 26,229 unique tracks:
1.  **Balanced Energy Wins:** Tracks with extreme energy (above 1 standard deviation) are actually **5.48 points less popular** on average than the baseline.
2.  **Volume vs. Quality:** Among the top 10 most prolific artists (like David Guetta and The Chainsmokers), track volume does not guarantee hits. Quality remains independent of quantity.
3.  **The Radio-Ready Cluster:** The **Pop** genre dominates the "Potential Hits" category, accounting for nearly 33% of tracks that meet high popularity and danceability thresholds.

---

## Technical Stack
*   **Language:** Python
*   **Data Manipulation:** Pandas
*   **Mathematical Operations:** NumPy (Manual vectorization and broadcasting)
*   **AI Tools:** Google Gemini / GitHub Copilot (Documentation & Debugging)

## How to Run
The analysis is encapsulated in the `SpotifyAnalyzer` class. To run the full pipeline:

```python
# initialize the analyzer with the dataset url
analyzer = spotifyanalyzer(url)

# execute the modular pipeline
analyzer.clean_data()
analyzer.process_numpy_analysis()

# generate a comprehensive json-ready report
print(analyzer.generate_report())
```

---
*Developed as part of the GDGOC AI/ML Learning Path.*
