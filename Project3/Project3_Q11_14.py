# QUESTION 11 
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader
import pandas as pd
import numpy as np
from surprise.model_selection import KFold
reader = Reader(rating_scale = (0.5,5))
df_ratings = pd.read_csv("ratings.csv")
surprises = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], reader)

receiver = df_ratings.pivot_table('rating', 'userId', 'movieId')
mean_of_users = np.mean(receiver, axis=1)
Rmatrix = df_ratings.pivot_table('rating', 'userId', 'movieId', fill_value=0)

reader = Reader(rating_scale = (0.5,5))
df_ratings_popular_surprise = Dataset.load_from_df(df_ratings_popular[['userId','movieId','rating']], reader)

reader = Reader(rating_scale = (0.5,5))
df_ratings_unpopular_surprise = Dataset.load_from_df(df_ratings_unpopular[['userId','movieId','rating']], reader)

reader = Reader(rating_scale = (0.5,5))
df_ratings_high_var_surprise = Dataset.load_from_df(df_ratings_high_var[['userId','movieId','rating']], reader)
RMSE_K_FoldPopular = []
RMSE_K_Fold = []
RMSE_Unpop = []
RMSE_HV = []
kf = KFold(n_splits=10)
for train, test in kf.split(surprises):
    predictions = [mean_of_users[i[0]] for i in test]
    true = [i[2] for i in test]
    RMSE_K_Fold.append(np.sqrt(mean_squared_error(true, predictions)))
for train, test in kf.split(df_ratings_popular_surprise):
    predictions = [mean_of_users[i[0]] for i in test]
    true = [i[2] for i in test]
    RMSE_K_FoldPopular.append(np.sqrt(mean_squared_error(true, predictions)))
for train, test in kf.split(df_ratings_unpopular_surprise):
    predictions = [mean_of_users[i[0]] for i in test]
    true = [i[2] for i in test]
    RMSE_Unpop.append(np.sqrt(mean_squared_error(true, predictions)))
for train, test in kf.split(df_ratings_high_var_surprise):
    predictions = [mean_of_users[i[0]] for i in test]
    true = [i[2] for i in test]
    RMSE_HV.append(np.sqrt(mean_squared_error(true, predictions)))
RMSE = np.mean(RMSE_K_Fold)
RMSE_Popular = np.mean(RMSE_K_FoldPopular)
RMSE_Unpopular = np.mean(RMSE_Unpop)
RMSE_High_Variance = np.mean(RMSE_HV)
print("Root Mean Squared Error for Movies: ", RMSE)
print("Root Mean Squared Error for Popular Movies: ", RMSE_Popular)
print("Root Mean Squared Error for Unpopular Movies: ", RMSE_Unpopular)
print("Root Mean Squared Error for High Variance Movies: ", RMSE_High_Variance)

#QUESTION 12
from surprise.model_selection import train_test_split
threshold = 3

FPR = []
TPR = []
AUC = []
threshold = 3
trainset, testset = train_test_split(surprises, test_size=0.1)
# From question 4 and 5, we observe that k = 20
knn = KNNWithMeans(k=20, sim_options={'name': 'pearson'})
knn.fit(trainset)
knnPred = knn.test(testset)
# From question 4 and 5, we observe that k = 20
nmf = NMF( n_factors = 20, n_epochs = 20, lr_bu = 0.007, lr_bi = 0.007, random_state = 42, verbose = False )
nmf.fit(trainset)
nmfPred = nmf.test(testset)
# Best SVD on all data previously found to be k=24 using cross validation
svd = SVD( n_factors = 24, n_epochs = 20, lr_all = 0.007, random_state = 42, verbose = False )
svd.fit(trainset)
svd_estimate = svd.test(testset)
GT = []
pred = []
for i in range(len(knnPred)):
    pred.append(knnPred[i].est)
    if testset[i][2] >= threshold:
        GT.append(1.0)
    else:
        GT.append(0.0)

fpr_knn, tpr_knn, thresholds = roc_curve(GT, pred)
AUC_knn = roc_auc_score(GT, pred)

GT = []
pred = []
for i in range(len(nmfPred)):
    pred.append(nmfPred[i].est)
    if testset[i][2] >= threshold:
        GT.append(1.0)
    else:
        GT.append(0.0)

fpr_nmf, tpr_nmf, thresholds = roc_curve(GT, pred)
AUC_nmf = roc_auc_score(GT, pred)
GT = []
pred = []
for i in range(len(svd_estimate)):
    pred.append(svd_estimate[i].est)
    if testset[i][2] >= threshold:
        GT.append(1.0)
    else:
        GT.append(0.0)

fpr_svd, tpr_svd, thresholds = roc_curve(GT, pred)
AUC_svd = roc_auc_score(GT, pred)

FPR.append(fpr_knn)
FPR.append(fpr_nmf)
FPR.append(fpr_svd)


TPR.append(tpr_knn)
TPR.append(tpr_nmf)
TPR.append(tpr_svd)

AUC.append(fpr_knn)
AUC.append(fpr_nmf)
AUC.append(fpr_svd)

# %% Ploting the ROC curves for all approaches
plt.figure(figsize=(15, 10))
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve', fontsize=23)
model_names = ['knn', 'nmf', 'mf']
plt.plot(FPR[0], TPR[0], label='ROC: Model: KNN AUC: {auc}'.format(auc=AUC[0]))
plt.plot(FPR[1], TPR[1], label='ROC: Model: NMF AUC: {auc}'.format(auc=AUC[1]))
plt.plot(FPR[2], TPR[2], label='ROC: Model: MF AUC: {auc}'.format(auc=AUC[2]))

plt.legend()
plt.show()


#QUESTION 14
threshold = 3
ts = [i for i in range(1, 25 + 1, 3)]

knn_precs = []
knn_recs = []
svd_precs = []
svd_recs = []
nmft_precs = []
nmft_recs = []
kf = KFold(n_splits=10)
bestSVD = 24
bestK = 20
bestNMF = 14
kf = KFold(n_splits=2)

for t in ts:
    precs_list_KNN = []
    recall_list_KNN = []
    precs_list_SVD = []
    recall_list_SVD = []
    precs_list_NMF = []
    recall_list_NMF = []

    for trainset, testset in kf.split(ratingsSurpriseSet):
        svd = SVD(n_factors=bestSVD, verbose=False)
        svd.fit(trainset)
        above_threshold = thresholding(testset, t, threshold)
        preds = svd.test(above_threshold)

        mean_prec_SVD, mean_rec_SVD = prec_rec(preds, t, threshold=3)

        precs_list_SVD.append(mean_prec_SVD)
        recall_list_SVD.append(mean_rec_SVD)

        knn = KNNWithMeans(k=bestK, sim_options={'name': 'pearson'})
        knn.fit(trainset)
        above_threshold = thresholding(testset, t, threshold)
        preds = knn.test(above_threshold)

        mean_prec_KNN, mean_rec_KNN = prec_rec(preds, t, threshold=3)

        precs_list_KNN.append(mean_prec_KNN)
        recall_list_KNN.append(mean_rec_KNN)

        nmf = NMF(n_factors=bestNMF, verbose=False)
        nmf.fit(trainset)
        above_threshold = thresholding(testset, t, threshold)
        preds = nmf.test(above_threshold)

        mean_prec, mean_rec = prec_rec(preds, t, threshold=3)

        precs_list_NMF.append(mean_prec)
        recall_list_NMF.append(mean_rec)

    nmft_precs.append(np.mean(precs_list_NMF))
    nmft_recs.append(np.mean(recall_list_NMF))

    svd_precs.append(np.mean(precs_list_SVD))
    svd_recs.append(np.mean(recall_list_SVD))

    knn_precs.append(np.mean(precs_list_KNN))
    knn_recs.append(np.mean(recall_list_KNN))

#
# plt.figure()
# plt.scatter(ts,knn_precs, s=30)
# plt.title("Precision vs t for knn")
# plt.show()

plt.figure()
plt.scatter(ts, knn_recs, s=30)
plt.title("Recall vs t for knn")
plt.show()

plt.figure()
plt.scatter(knn_recs, knn_precs, s=30)
plt.title("Precision vs Recall for knn")
plt.show()

plt.figure()
plt.scatter(ts, nmft_precs, s=30)
plt.title("Precision vs t for NMF")
plt.show()

plt.figure()
plt.scatter(ts, nmft_recs, s=30)
plt.title("Recall vs t for NMF")
plt.show()

plt.figure()
plt.scatter(nmft_recs, nmft_precs, s=30)
plt.title("Precision vs Recall for NMF")
plt.show()

plt.figure()
plt.scatter(ts, svd_precs, s=30)
plt.title("Precision vs t for MF with bias")
plt.show()

plt.figure()
plt.scatter(ts, svd_recs, s=30)
plt.title("Recall vs t for MF with bias")
plt.show()

plt.figure()
plt.scatter(svd_recs, svd_precs, s=30)
plt.title("Precision vs Recall for MF with bias")
plt.show()

plt.figure()
plt.scatter(knn_recs, knn_precs, s=30, label="k-NN")
plt.scatter(nmft_recs, nmft_precs, s=30, label="NMF")
plt.scatter(svd_recs, svd_precs, s=30, label="MF with bias")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall for different models")
plt.legend()
plt.show()
