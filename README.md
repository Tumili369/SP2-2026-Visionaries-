# SP2-2026-Darks
# Sona Power Predict - 2026
**College Name:** Sona College of Technology  
**Team Name:** Visionaries  

### Team Members
* **TUMILI** - Year 2, Computer Science and Design (CSD)
* **SUBHASHREE** - Year 2, Computer Science and Design (CSD)
* **DHURGADEVI** - Year 2, Computer Science and Design (CSD)
* **SUPRIYA** - Year 2, Computer Science and Design (CSD)

---

### Overview

An IPL Powerplay score prediction model that estimates the total runs scored during overs 1–6 of an IPL innings. Built with **XGBoost** and trained on ball-by-ball IPL data from 2008 to 2025, with recency weighting to reflect modern scoring patterns.

| Property | Detail |
|---|---|
| **Task** | Regression — predict powerplay run total per innings |
| **Algorithm** | XGBoost (`XGBRegressor`) |
| **Training window** | IPL 2019 – 2025 (recency-weighted) |
| **Input features** | 8 (team averages, player SR & economy, venue, innings) |
| **Interface** | `fit(deliveries_df, players_df, matches_df)` → `predict(test_df)` |

---

### Required Data Files

| File | Key Columns |
|---|---|
| `deliveries_updated_ipl_upto_2025.csv` | `matchId`, `date`, `over`, `batting_team`, `bowling_team`, `batsman`, `bowler`, `batsman_runs`, `extras` |
| `ipl_players_uniqueid.csv` | `ID`, `Player_Name`, `Team` |
| `matches_updated_ipl_upto_2025.csv` | `matchId`, `venue` |

---

### How It Works

**1. Data Preparation**  
Ball-by-ball deliveries are filtered to powerplay overs (`over < 6`). Team names and venue names are normalised automatically across all seasons (e.g. *"Delhi Daredevils" → "Delhi Capitals"*, *"Feroz Shah Kotla" → "Arun Jaitley Stadium"*) — no names are hardcoded; all mappings are derived at runtime from the supplied DataFrames.

**2. Player Name Resolution (4-Pass Alias Builder)**  
IPL delivery data uses abbreviated names (e.g. `"RG Sharma"`) while the players reference CSV uses full canonical names (`"Rohit Sharma"`). A 4-pass algorithm resolves this automatically:

| Pass | Strategy |
|---|---|
| 1 | Direct full-name match |
| 2 | Initials + surname prefix match |
| 3 | Team-filtered disambiguation when initials are ambiguous |
| 4 | Surname + team + first-initial fallback |

**3. Feature Engineering (8 features)**

| # | Feature | Description |
|---|---|---|
| 0 | `team_bat_avg` | Weighted PP average of the batting team (2019+) |
| 1 | `team_bowl_avg` | Weighted PP average of the bowling team (2019+) |
| 2 | `avg_bat_sr` | Mean strike rate of the batting lineup |
| 3 | `top_bat_sr` | Best strike rate in the batting lineup |
| 4 | `avg_bowl_econ` | Mean economy rate of the bowling lineup |
| 5 | `best_bowl_econ` | Best (lowest) economy rate among bowlers |
| 6 | `innings` | 1st or 2nd innings |
| 7 | `venue_pp_avg` | Historical PP average at the venue |

**4. Recency Weighting**  
Training is restricted to 2019 onwards. Seasons are further weighted to emphasise modern IPL scoring rates:

| Season | Weight |
|---|---|
| 2024 – 2025 | 6× |
| 2023 | 4× |
| 2022 | 3× |
| 2021 | 2× |
| 2020 | 1.5× |
| 2019 | 1× |

**5. Calibration Offset**  
A data-derived calibration offset is computed after training — the gap between the weighted mean of actual 2022–2025 powerplay scores and the model's mean prediction on that slice. This is added at predict time to align outputs with modern IPL scoring (~55–62 runs). No constant is hardcoded; the offset is fully derived from the data.

---

### Output Format

`predict()` returns a DataFrame with two columns:

| Column | Type | Description |
|---|---|---|
| `id` | int | Row identifier from `test_df` |
| `predicted_score` | int | Predicted powerplay run total |

---

### Libraries Used in Model

Based on the `mymodelfile.py` submission, the following Python libraries are utilized for data manipulation and mathematical operations:

* **`pandas`**: Used for DataFrame manipulation, grouping, and handling ball-by-ball historical IPL data.
* **`numpy`**: Used for numerical operations and array handling.
* **`xgboost`**: Core gradient-boosted regression model (`XGBRegressor`) used for training and prediction.
* **`scikit-learn`**: Used for train/test splitting (`train_test_split`) and validation scoring (`mean_absolute_error`).
* **`re`**: Used for parsing player ID strings from test data.
* **`os`**: Standard library utility import.

---

### License

This project is licensed under the **MIT License**.
