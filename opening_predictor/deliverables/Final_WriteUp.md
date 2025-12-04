# Contributors: Kannan Jain & Rachana Raju

## Motivation:
People are increasingly saying that chess games are becoming more and more memorized. This is hard to believe given that there are ~400 different moves a game can take within the first full move. This number only increases with each move thereafter so it’s impossible to memorize every single game possible. However a common strategy is for players to learn different openings which are the first 5-10 moves a player plays in the game. These initial moves determine how the board develops as the game progresses. Additionally many players practice by solving various exercises called chess puzzles. These exercises usually present players with a board in the middle game where one side has an advantage and the player brainstorms the best way to take advantage of the board position. A player may want to practice opening lines that may lead to these puzzle-like positions advantageous to the color they’re playing as. This was the motivation behind building our chess opening classifier. 

## Data retrieval, cleaning, and feature engineering:
The data source is [this kaggle dataset](https://www.kaggle.com/datasets/adityajha1504/chesscom-user-games-60000-games/data) with over 60,000 games from chess.com. The goal of this model is to identify an opening origin given a middle game chess board position. This analysis uses the board position after 10 moves as a middle game position. For this analysis, 10 full moves were chosen as the middle game point because the majority of the audience on chess.com are not professional players who practice extended opening variation up to 10-12 moves but instead players who play chess as a hobby and deviate from an opening after 5-6 moves. Our goal was to analyze early middle games so that a potential puzzle position can be tied back to an opening.

The dataset had information about an opening as EcoName and EcoCodes. EcoCodes were used to group variations on one big opening as an opening family. Here are the mappings used: https://www.365chess.com/eco.php. Using these, the opening families with the highest number of games played (the most popular openings with more than 1000 games) were identified. 

Next, the PGN, a sequence of chess moves, was utilized to extract board positions in a format called FEN at every single move up until move 15. This was done because we were not sure if we wanted to feed a sequence of FENs (board positions) such as moves 10-12 to our model or just the FEN (position at a single move). While the dataset had an FEN column, it wasn’t helpful for this use case because that FEN came from the position the game was ended at. Thus, creating our own FENs was a crucial step.

In order to embed the FEN in the chess board, a 3d array representing each unique piece’s position on the chess board was constructed. The 3d array, consisting of 12 8x8 matrices was flattened into a 1d array of 768 features to be fed into the model


## Creating Train and Test Split

The train and test split was created by randomly choosing 80 percent of each opening family’s occurrences to be in the training set. The remaining 20 percent for each opening family was used for the test set.

## Additional Features:
Material points for both white and black were later added as new features (in addition to the previous 768 features to represent the board itself). This is a standardized calculation used in chess as each piece has some value points. For example, each Rook is worth 5 points, Knight is worth 3, Bishop is worth 3, Queen is worth 9, and Pawn is worth 1. These were summed up for white and black individually and used as additional features. Precision and recall improved by about 1% each by adding material points. 
An additional feature that was tried was using stockfish, a chess engine’s evaluation, as a feature. That only helped raise the precision and recall by ~0.7% which did not justify its compute and added model complexity. 

## Model Results
Here are the results for the classification of each class using the 770 features on a Random Forest (RF) Model. Random Forest was chosen as the model to be used after first experimenting with both RF and a K-Nearest Neighbors model. The previous comparisons can be found in model_eval.md. 

```
Class                          Precision    Recall       F1-Score    
------------------------------------------------------------------
Bishop's Opening               0.6065       0.3013       0.4026      
Caro-Kann Defence              0.8123       0.7084       0.7568      
Centre Game                    0.8609       0.5000       0.6326      
English Opening                0.8073       0.5016       0.6188      
French Defence                 0.6498       0.6016       0.6247      
Italian Game                   0.5770       0.7630       0.6571      
King's Pawn Opening            0.5630       0.7692       0.6502      
Modern Defence (Robatsch)      0.6429       0.5878       0.6141      
Nimzovich-Larsen Attack        0.7157       0.6498       0.6812      
Petrov's Defence               0.7770       0.4355       0.5581      
Philidor's Defence             0.6545       0.6687       0.6616      
Pirc Defence                   0.6212       0.4141       0.4970      
Polish Opening                 0.7497       0.6170       0.6769      
Queen's Gambit                 0.7871       0.8007       0.7938      
Queen's Pawn                   0.7349       0.8525       0.7894      
Reti Opening                   0.7294       0.2441       0.3658      
Ruy Lopez                      0.8818       0.8739       0.8778      
Scandinavian Defence           0.7600       0.7972       0.7782      
Sicilian Defence               0.8252       0.9707       0.8921      
Two Knights Defence            0.7396       0.6068       0.6667      
Vienna Game                    0.7981       0.3689       0.5046      
------------------------------------------------------------------
Accuracy: 0.7136    
Overall Precision: 0.7207
Overall Recall: 0.7136
```
Looking at this, for most of the openings, there’s a good balance of both precision and recall and most f1-scores are above 0.6.

Looking at the [PR-AUC curve](../pr_curve_final.png) the model had strong performance in classifying all openings. Even the openings with lower PR-AUC curves were still much better than randomly assigning an opening. Random assignment of an opening would lead to a precision of 0.0476 and all classes had a curve significantly higher than 0.048 for most recall values. This demonstrates that the model actually learned how to classify openings.

Taking a closer look at the classes that didn't have a good precision and recall balance, it was noticed that they appeared in the dataset less frequently than other openings. Although an 80/20 split for each opening family was used to have a good representation of each family in the training set, the amount of times each opening showed up in the training dataset differed depending on its frequency in the overall dataset ranging from 9713 to 1025. 

```
Class                          Precision    Recall       F1-Score    Dataset Occurrences   
-------------------------------------------------------------------------------------------
Bishop's Opening               0.6065       0.3013       0.4026         1673
Centre Game                    0.8609       0.5000       0.6326         1025
English Opening                0.8073       0.5016       0.6188         1605
Petrov's Defence               0.7770       0.4355       0.5581         1310
Pirc Defence                   0.6212       0.4141       0.4970         1557
Reti Opening                   0.7294       0.2441       0.3658         1340
Vienna Game                    0.7981       0.3689       0.5046         1181
-------------------------------------------------------------------------------------------
```

Other interesting observations such as low evaluation scores for similar openings were noticed. For example, English Opening and Reti Opening are very similar openings in the sense that they both follow the same general tactic of trying to occupy the same cells with the same pieces early on in the game. 

Given that the model was classifying 21 different chess openings, an overall precision and recall scores of 0.7 demonstrate that our classification model has strong performance and is able to effectively classify most classes. 

## Future Work
In the future we would like to spend more time on stockfish evaluation and using that as a feature. We suspect that there is definitely more value to it once we explore it more which can also help with dimensionality reduction. Right now, when dimensionality reduction was attempted, by emphasizing more on pawn structures, it was found to be quite tedious. However, a tool such as stockfish would already take those things into account. Given a bigger dataset with good representation we would want to find specific popular variation of an opening. Currently the variation that has the highest occurrence frequency is Bishop's Opening with 1206. However, given the split using opening families those are the openings with the lowest occurrence frequency in the dataset used by the model. 