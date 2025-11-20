## Feature Engineering
We are building an opening classifier based on what the board looks like after 10 full moves (a full move is when both sides move). To do this we had to use the pgn column that was in our dataset and build the board for each move that a player made and after 10 full moves, record the resulting FEN (which is a string used to record what the board looks like after a certain move). We did this because the FEN column that existed in the original dataset was just the FEN after the game ended which is not what we wanted. Then with the FENS we built a 3d array (we represented the FEN as 12 8x8 matrices since there are 6 chess pieces but we need to represent the different sides so we get 12 and we want to show where each piece is on the 8 x 8 board).

Then to create the matrix to feed into our model, we needed a 1d array. In order to do that we had to flatten our 3d array into a 1d array of 768 dimensions. We added no other features, just the 768 columns that represented where each piece was on the board. We used random forest and knn and then plotted the PR and ROC curves. 

## Evaluation

### RF Model
![Alt text](roc_curve.png)
![Alt text](pr_curve.png)
Looking at the ROC curve of one vs rest of each classification for random forest, we can see that each classification was better than random. Looking at the PR-AUC for one vs rest for random forest we can see that our model was better in the classification of some openings compared to others. 

This is the feature importance that we got from our random forest model
e4         0.004145      
e3         0.004143     
d4         0.004103     
c7         0.003928      
c4         0.003088    
e5         0.003018       
e2         0.002790       
c5         0.002728    
e7         0.002697      
d2         0.002370        
d5         0.002236 

From this we see that distinctive pawn moves like e4, d4, c5, e5, that mark the beginning of popular openings lines like bishops opening, Sicilian defense, queen’s gambit, queens pawn had the highest feature importance. This adds up to our expectations because these pawn states are pretty consistent up until middle game. We also noticed that the highest ranked features were based on pawn placements.

### KNN Model
Our knn model did not do as well as our random forest model. Looking at the PR and ROC curve we can see that it’s clearly worse than the curves we had from random forest. However we can still see the same trends for which openings our model was able to classify better than others, for both random forest and knn classification of Sicilian Defence was one of the best while Reti Opening was the worst.
![Alt text](knn_pr_curve.png)
![Alt text](knn_roc_curve.png)

## Ideas for final report

We did a 80/20 train test split for each opening but looking at the opening that had the best pr curve we noticed that it was an opening that had the 2nd most appearances in our dataset. This could mean that the models that appeared more times in our training test were able to be better classified so one thing we might do in the future is to use the same amount of occurrences of each opening in both our train and test data.

Some ideas we have for our final report is to see if we can do any dimensionality reduction. Obviously 768 dimensions is a lot and could be limiting the performance of our classifier. We will try to see if we can use something like SVD to get a smaller matrix to feed into our model. Another thing we are looking into is possibly using other models like XG Boost to see if that could also be another source of improvement. We are also considering other features not related to where pieces are located but other signals like material advantage, isolated pawns, and double pawns.