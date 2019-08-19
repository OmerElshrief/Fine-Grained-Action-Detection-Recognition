from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle


def evaluate_average_percision(Y_test, y_score, n_classes):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    return precision,recall,average_precision
    
def print_Average_precision(precision, recall, average_precision, n_classes, plot_classes = False):
    
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b'    )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    
    if plot_classes:
      colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','teal'])
      plt.figure(figsize=(7, 8))
      f_scores = np.linspace(0.2, 0.8, num=4)
      lines = []
      labels = []
      for f_score in f_scores:
          x = np.linspace(0.01, 1)
          y = f_score * x / (2 * x - f_score)
          l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
          plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

      lines.append(l)
      labels.append('iso-f1 curves')
      l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
      lines.append(l)
      labels.append('micro-average Precision-recall (area = {0:0.2f})'
                    ''.format(average_precision["micro"]))

      for i, color in zip(range(n_classes), colors):
          l, = plt.plot(recall[i], precision[i], color=color, lw=2)
          lines.append(l)
          labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                        ''.format(i, average_precision[i]))

      fig = plt.gcf()
      fig.subplots_adjust(bottom=0.25)
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title('Extension of Precision-Recall curve to multi-class')
      plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
      
      
def evaluation(on_hot_predictions, ground_truth):
  """"
  input: one hot encoded predictions, ground truth labels 
  output: list of 5 precisions for each class 
  """""
  
  predictions = np.argmax(on_hot_predictions, axis = 1)
  pred = []
  for idx, _ in enumerate(predictions): 
    pred.append(np.dot(np.where(predictions == idx), 6))
  
  labels = []
  for category in ground_truth:
    labels.append(list(map(lambda x: list(range(x[0],x[1])), category[0])))
  
  true = []
  for idx, _ in enumerate(ground_truth):
    true.append(list(map(lambda x: np.sum(np.isin(pred[idx][0], x)), labels[idx])))
  
  true_positives = []
  for idx, _ in enumerate(ground_truth):
    true_positives.append(np.sum(np.array(true[idx]) != 0))
    
  false_positives = []
  for idx, _ in enumerate(ground_truth):
    false_positives.append(len(pred[idx][0]) - true_positives[idx])
  
  precison = np.array(true_positives) / (np.array(true_positives) + np.array(false_positives))
  return precison 






# This is an implementation of the mAP metric calculation over classes 1-5 (ignoring class 0)

def evaluate(result, gt):
  assert(len(result) == len(gt)), 'input arrays must have the same size'
  
  # saving true positives (tp) counts and false positives (fp) counts
  tp = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0}
  fp = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0}
  
  for index in range(0, len(gt)):
    # ignoring class 0
    if gt[index] != 0 or result[index] != 0:
      if gt[index] == result[index]:
        tp[str(gt[index])] += 1
      else:
        fp[str(result[index])] += 1

  # calculating precision per class as tp/(tp + fp)
  AP = {k : tp[k]/(tp[k] + fp[k]) for k in tp if k in fp}
  print ('precision per class =', AP)
  mAP = float(sum(AP.values())) / len(AP)
  print ('mAP =', mAP)
  return mAP
      
