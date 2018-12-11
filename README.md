# ml-kaggle-competition
Codes for the Shopee-IET Machine Learning Kaggle Competition (Achieved rank 2 out of 51)

Contest page link: https://www.kaggle.com/c/shopee-iet-machine-learning-competition#description
Leaderboard link (team name - hesl): https://www.kaggle.com/c/shopee-iet-machine-learning-competition/leaderboard

Used a deep learning based ensemble model and a few data augmentation techniques to obtain top performance in the above challenge.

Approach is explained in the presentation.pptx file

Prerequisites:
1. Python 3
2. Pytorch 0.3

Main scripts:
1. train.py - Trains a model on the given data
2. train_top5.py - Writes the top 5 predictions with their confidence for trained model on the training data.
3. test.py - tests the model and outputs predictions
4. test_top5.py - Predicts the top 5 classes along with their confidence on test data.
5. ensemble_optimal.py - Uses the top 5 predictions of each model to give final prediction