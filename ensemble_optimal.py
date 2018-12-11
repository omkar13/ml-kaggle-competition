"""
Perform an ensemble of the trained model predictions
"""

import numpy as np
import pandas as pd
import os

resnet = np.load('./conf_mats/resnet_conf_mat.npy')
xception = np.load('./conf_mats/xception97_conf_mat.npy')
inceptionv4 = np.load('./conf_mats/inceptionv4_88_conf_mat.npy')
inceptionv3 = np.load('./conf_mats/inceptionv3_91_conf_mat.npy')

resnet_classwise = np.max(resnet, axis=1) / np.sum(resnet, axis=1, dtype=np.float32)
xception_classwise = np.max(xception, axis=1) / np.sum(xception, axis=1, dtype=np.float32)
inceptionv4_classwise = np.max(inceptionv4, axis=1) / np.sum(inceptionv4, axis=1, dtype=np.float32)
inceptionv3_classwise = np.max(inceptionv3, axis=1) / np.sum(inceptionv3, axis=1, dtype=np.float32)
resnet101_classwise = np.load('./conf_mats/accuracy_resnet101.npy')

data_dir = './top5'
resnet50_dir = os.path.join(data_dir, 'test_data/resnet')
inceptionv4_dir = os.path.join(data_dir, 'test_data/inception')
inceptionv3_dir = os.path.join(data_dir, 'inceptionv3_test_top5')
xception_dir = os.path.join(data_dir, 'test_data/xception')
resnet101_dir = os.path.join(data_dir, 'resnet101_test_top5')

def calculate_normalized_scores(top5_dir, classwise_score):
    names = []
    norm_scores = []
    print('Reading ', top5_dir)
    for i in range(1, 16111 + 1):
        df = pd.read_csv(os.path.join(top5_dir, str(i) + '.csv'))
        arr = df.as_matrix()
        confidence_hash = np.zeros(18)
        for class_confidence in arr:
            class_id = int(class_confidence[0])
            confidence = class_confidence[1]
            confidence_hash[class_id] = confidence

        normalized_score = confidence_hash * classwise_score
        names.append(str(i))
        norm_scores.append(normalized_score)

    return norm_scores, names



#resnet50_scores, names = calculate_normalized_scores(resnet50_dir, resnet_classwise)
xception_scores, names = calculate_normalized_scores(xception_dir, xception_classwise)
inceptionv4_scores, names = calculate_normalized_scores(inceptionv4_dir, inceptionv4_classwise)
inceptionv3_scores, names = calculate_normalized_scores(inceptionv3_dir, inceptionv3_classwise)
resnet101_scores, names = calculate_normalized_scores(resnet101_dir, resnet101_classwise)

ensemble = np.add(np.add(np.add(resnet101_scores, inceptionv4_scores), xception_scores), inceptionv3_scores)
final_score = []

for i in range(len(ensemble)):
    final_score.append([names[i], np.argmax(ensemble[i])])
final_score = np.array(final_score)
df = pd.DataFrame(final_score, columns=['id', 'category'])
df.to_csv('abhinandpo_wins_again.csv', index=False)

'''
resnet_mask = np.zeros(18)
inceptionv4_mask = np.zeros(18)
xception_mask = np.ones(18)

for i in [6, 13, 16]:
    resnet_mask[i] = 1
    xception_mask[i] = 0
for i in [0, 3, 5, 10, 14, 17]:
    inceptionv4_mask[i] = 1
    xception_mask[i] = 0

mask_dict = {'resnet50_83.csv' : [resnet_mask, x], 'xception_86.csv' : [xception_mask, y], 'deeper_trained_inceptionv4_84.csv': [inceptionv4_mask, z]}



final_score =[]
count = 0
count2 = 0
for i in range(len(csv_list[0][:,0])):
    scores_and_preds = []
    predictions = []
    for j, arr in enumerate(csv_list):
        pred = arr[i, 1]
        predictions.append(pred)
        #print(pred, names[j])
        if mask_dict[names[j]][0][pred] == 1:
            model_score = mask_dict[names[j]][1][pred]
            best_diff = 1000
            for model in names:
                diff = model_score - mask_dict[model][1][pred]
                minimum = min(diff, best_diff)
                best_diff = minimum if minimum > 0.0 else best_diff
            scores_and_preds.append([pred, best_diff])


    #print(scores_and_preds)
    final_class = -1
    if len(scores_and_preds) == 1:
        best_score = 100
        for pair in scores_and_preds:
            if pair[1] < best_score:
                best_score = pair[1]
                final_class = int(pair[0])
    else :
        final_class = max(predictions, key=predictions.count)
        #print(i, len(scores_and_preds), final_class)


    final_score.append([i+1, final_class])

#print(count2)
final_score = np.array(final_score)
df = pd.DataFrame(final_score, columns=['id', 'category'])
df.to_csv('optimal_ensemble_democracy.csv', index=False)
#not_eq = csv_list[5][1:, 1] != final_score[1:, 1]
results = []

compare_model = csv_list[0]
compare_model2 = csv_list[1]
compare_model3 = csv_list[2]

for i in range(len(final_score[:,1])):
    if compare_model[i, 1] != final_score[i, 1] or compare_model2[i, 1] != final_score[i, 1] or compare_model3[i, 1] != final_score[i, 1]:
        results.append([i + 1, compare_model[i, 1], compare_model2[i, 1], compare_model3[i, 1], final_score[i, 1]])
        #print('imagenum ', i + 1, best_model[:-4], compare_model[i, 1], 'ensemble', final_score[i, 1])

df = pd.DataFrame(np.array(results), columns=['image_id', 'resnet50_83', 'xception_86', 'inceptionv4_84', 'ensemble'])
df.to_csv('optimal_ensemble_changes-incep83-resnet83-xcep86.csv', index=False)
#print(not_eq)
print(np.sum(compare_model2[:, 1] == final_score[:, 1]) / 16111.0)
'''
