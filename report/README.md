# Model Selection and Challenges

The project aims to build a robust recommendation system, exploring two models:
1. A Neural Network based on embedded item and user information.
2. An Alternating Least Squares model using matrix factorization techniques.

## Resources used:
1. Google + ChatGPT + [Reddit](https://www.reddit.com/) - for filling knowlege gaps, like which libraries can be used with CUDA
2. ChatGPT - For speeding up writing code for well known tasks that do not require complex and non-trivial logic
3. [Stack Overflow](https://stackoverflow.com/) / [Stack Exchange](https://stackexchange.com/) - for adressing specific issues I encountered during implementation
4. Documentation of used libraries and frameworks - for usage reference and in search for functions

## Key challenges:
## 1. Big sparce dataset of user-item interactions  
One possible solutions is to organize `users` and `items` into groups. For this purpose embedded representations can be used together with fast and powerful [FAISS](https://ai.meta.com/tools/faiss/) library, or any other clustering algorithm (I actually tried this solution and found ~430 groups for both `users` and `items`).   
Another solution can be to filter out unpopular `items` (<10 reviews) and inactive `users` (<5 reviews). Further steps will be arain use similarity search to map incative customers to the the most similar active ones. If the same recommendation will appeat too often for different users, a random choice from top 20 the most similar products can be used.

## 2. Mostly textual data in the dataset   
This was addressed by two different ways:
    - Embedded textual information for Neural Network Model
    - Matrix Factorization Model that does not require any additional information, besides the IDs.

## 3. Unbalanced data with one major and 4 minor groups.  
This was addressed by binarizing the target. The idea here is that we want to recommend only products that are likely to get the highest rating, thus only reviews with rating 5 were marked as positive. This solution is not perfect, since there will definitely be similar reviews with rating 5 and 4, which will confuse a model during its training. Besides that (the dataset still stayed unballanced) the random undersampling technique was applied ([RandomUnderSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html) from imblearn library). Possibly, a better approach would be to use [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html#smote) oversampling technique, but this appeared to be extremely computationally intensive for the dataset.   
Possible improvements are:
    - collect more data, especially with 'negative' reviews
    - completely drop reviews with reting 4 to create bigger difference between positive and negative reviews.

## 4. Cold start problem, when recommendations needs to be provided for unknown users (no user profile).   
This was solved by applying the following logic:  
    - the dataset was limited to only 30 last days (to capture trends, season, fashion patterns)
    - only top rated items were selected
    - top 5 popular items were used as cold start recommendations.

## 5. Computationally expensive prepocessing of textual information. 
This mostly affected Neural Network Model which used three sets of embeddings. Additionally the `price` column was used as an input and to impute missing values [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) from scikit-learn library was used. The only solution here is to use machine with GPU to make embedding process faster.

The chosen model, a Neural Network, demonstrates better results in terms of AUC and adaptability for future improvements, such as integrating external data sources to enhance the recommendation quality.
