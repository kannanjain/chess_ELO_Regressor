# EDA Summary: Chess.com Games Dataset

## What is your dataset and why did you choose it?

We selected the **Chess.com User Games dataset** available on Kaggle (https://www.kaggle.com/datasets/adityajha1504/chesscom-user-games-60000-games). This dataset contains over 60,000 chess games played on Chess.com: a popular online chess platforms with millions of active users.

### Dataset Structure

The dataset includes the following key features:

- **Player Information**: `white_username`, `black_username`, `white_id`, `black_id`
- **Rating Data**: `white_rating`, `black_rating` (ELO ratings)
- **Game Outcomes**: `white_result`, `black_result` (win, loss, draw, resignation, timeout, etc.)
- **Time Controls**: `time_class` (blitz, bullet, rapid, daily), `time_control` (format: total_time + increment)
- **Game Format**: `rules` (standard chess or variants like Chess960, crazyhouse), `rated` (whether ELO is at stake)
- **Game Records**: `pgn` (Portable Game Notation - complete move sequence), `fen` (Forsyth-Edwards Notation - board positions)

### Why This Dataset?

This dataset is ideal for our **classification project** because:

1. **Rich Feature Set**: The PGN (Portable Game Notation) contains the complete sequence of moves for each game, which we can parse to extract numerous features such as:
   - Move patterns and sequences
   - Time spent per move (embedded in PGN)
   - Opening strategies
   - Game complexity metrics

2. **Clear Classification Target**: Player ratings (ELO) provide a natural target variable for classification (e.g., beginner vs. intermediate vs. advanced) or regression

3. **Sufficient Volume**: With 60,000+ games, we have enough data to train robust machine learning models

4. **Data Quality**: Most games (~59,500+) contain valid, non-empty PGN records, ensuring we have the core data needed for our analysis


---

## What did you learn from your EDA?

### Key Findings

#### 1. **Time Control Distribution**
Most games are **blitz** format (3 minutes + 2 second increment per move). This was unexpected, as we initially anticipated seeing more standard time controls (like 1 hour classical games) or rapid games.

Thus, we will focus our classification model specifically on blitz game as different time controls change player decision-making and move quality.

#### 2. **Rating Distribution**
Player ratings follow an approximately normal distribution centered around 1200-1400 ELO, with:
- Range: ~400 to ~2400 ELO
- Mean: ~1300 ELO
- Standard deviation: ~300 points

This distribution is expected and aligns with how ELO systems naturally distribute players. It suggests we have good representation across skill levels, though we may need to address class imbalance if we bin ratings into categories (e.g., beginner/intermediate/advanced).

#### 3. **Rules Variety**
Nearly all games (~99%+) use standard chess rules rather than variants (crazyhouse, three-check, Chess960, etc.).

So, we can safely exclude variants from our model and focus on standard chess, simplifying feature engineering and model training.

#### 4. **Time Per Move**
Analysis of the PGN format revealed that each move includes timestamp data showing time remaining on the player's clock. This allows us to calculate:
- Time spent per move
- Clock management patterns

---

## What issues or open questions remain?

### 1. **Rating Ambiguity**
Chess.com maintains separate ratings for different time controls (blitz, rapid, bullet, classical). Our dataset shows a single rating per player per game, but it's unclear whether this rating corresponds to their blitz rating, rapid rating, or overall rating

We tried looking into player information from their API but did not see different rating categories in it 
```
{'player_id': 9142478, '@id': 'https://api.chess.com/pub/player/-amos-', 'url': 'https://www.chess.com/member/-Amos-', 'name': 'Nicholas Amos', 'username': '-amos-', 'followers': 0, 'country': 'https://api.chess.com/pub/country/US', 'location': 'Louisiana', 'last_online': 1360354745, 'joined': 1349387260, 'status': 'basic', 'is_streamer': False, 'verified': False, 'streaming_platforms': []}
```

### 2. **PGN Embedding Strategy**
 PGN is a text-based format containing sequential move data. We need to determine how to embed this data for machine learning.

We want to explore:
- Parse PGN to extract numerical features (number of moves, capture rate, piece activity)
- Use chess engines (like Stockfish) to evaluate position quality after each move
- Apply sequence models (LSTM, Transformer) to learn from move sequences directly
- Create opening features by mapping early moves to known opening theory
- Extract temporal features (time spent per move, time pressure situations)

**Next Steps**: Research existing chess AI literature for feature engineering approaches

### 3. **Opening Diversity Feature**
Create a feature measuring "opening diversity" to capture how varied a player's opening choices are.

**Challenge**: We need a mapping from opening moves (or FEN positions) to standard opening names (e.g., "Sicilian Defense," "Queen's Gambit").

**Next Steps**:
- Search for chess opening databases or APIs (e.g., lichess opening database)
- Consider using the first 5-10 moves to classify openings
- Alternatively, use FEN positions to match against known opening book positions
---

## Summary

This dataset is well-suited for our classification project. Our EDA revealed important decisions (focus on blitz, standard rules) and uncovered valuable features (time per move). The main open questions revolve around feature engineering approaches for sequential move data and validation of rating-time control correspondence.