import numpy as np
import pandas as pd

def read_file(filename):
  df = pd.read_csv(filename)
  return np.array(df)

x_train = read_file('x_train.csv')
y_train = read_file('y_train.csv')
x_test = read_file('x_test.csv')
y_test = read_file('y_test.csv')
alpha = 5

def split_ones(x_train, y_train):
  return x_train[np.where(y_train == 0)[0]], x_train[np.where(y_train == 1)[0]]

def get_word_counts(x, y):
  zeros, ones = split_ones(x, y)
  word_counts = np.zeros((2, x.shape[1]))
  word_counts[0] = np.sum(zeros, axis=0)
  word_counts[1] = np.sum(ones, axis=0)
  return word_counts

def get_prior():
  # np.sum gives count of 1's
  return np.sum(y_train) / y_train.shape[0]

def get_likelihoods(x, word_counts, smooth):
  l = np.zeros((2, x.shape[1]))
  for i, row in enumerate(word_counts):
    for j, count in enumerate(row):
      if smooth:
        l[i,j] = (count + alpha) / (np.sum(row) + alpha * row.shape[0])
      else:
        l[i, j] = count / np.sum(row)

  l = np.log(l, where=(l!=0))
  l[l == 0] = -1 * pow(-10, 12)
  return l

def get_likelihoods_bernoulli():
  l = np.zeros((2, x_train.shape[1]))
  for i, row in enumerate(l):
    word_counts = np.count_nonzero(x_train[np.where(y_train == i),:][0], axis=0)
    label_count = y_train[y_train==i].shape[0]
    for j, col in enumerate(row):
      l[i,j] = word_counts[j] / label_count

  return l


def classify(smooth):
  word_counts = get_word_counts(x_train, y_train)
  l = get_likelihoods(x_train, word_counts, smooth)
  p_spam = get_prior()

  ham_A = x_test * l[0]
  spam_A = x_test * l[1]

  ham_predictions = np.log(1-p_spam) + np.sum(ham_A, axis=1)
  spam_predictions = np.log(p_spam) + np.sum(spam_A, axis=1)
    
  predictions = spam_predictions > ham_predictions

  return predictions


def classify_bernoulli():
  p_spam = get_prior()
  l = get_likelihoods_bernoulli()
  x_bern = (x_test > 0).astype(int)
  
  ham_log = x_bern * l[0] + (1 - x_bern) * (1 - l[0])
  ham_log[ham_log == 0] = -1 * pow(-10,12) 
  ham_predictions = np.log(1-p_spam) + np.sum(np.log(ham_log, where=(ham_log!=(-1 * pow(-10, 12)))), axis=1)

  spam_log = x_bern  * l[1] + (1 - x_bern) * (1 - l[1])
  spam_log[spam_log == 0] = -1 * pow(-10,12)
  spam_predictions = np.log(p_spam) + np.sum(np.log(spam_log, where=(spam_log!=(-1 * pow(-10, 12)))), axis=1)
    
  predictions = spam_predictions > ham_predictions
  return predictions


def test(smooth=False, bernoulli=False):
  if bernoulli:
    predictions = classify_bernoulli()
  else:
    predictions = classify(smooth)
  true_pos = 0
  false_pos = 0
  true_neg = 0
  false_neg = 0
  for i, label in enumerate(y_test):
    if predictions[i] == 1 and label == 1:
        true_pos += 1
    elif predictions[i] == 1 and label == 0:
        false_pos += 1
    elif predictions[i] == 0 and label == 1:
        false_neg += 1
    else:
        true_neg += 1        
            
  return (true_pos + true_neg) / ( true_pos + false_pos + true_neg + false_neg), [[true_pos, false_neg], [false_pos,true_neg]], false_pos + false_neg

if __name__ == '__main__':
  accuracy2, matrix2, wrong_count2 = test()
  print('-------Question 3.2-------')
  print('Accuracy: ', accuracy2)
  print('Confusion Matrix: ', matrix2)
  print('Total Wrong Predictions: ', wrong_count2)


  accuracy3, matrix3, wrong_count3 = test(smooth=True, bernoulli=False)
  print('-------Question 3.3-------')
  print('Accuracy: ', accuracy3)
  print('Confusion Matrix: ', matrix3)
  print('Total Wrong Predictions: ', wrong_count3)
  print('finished')


  accuracy4, matrix4, wrong_count4 = test(smooth=False, bernoulli=True)
  print('-------Question 3.4-------')
  print('Accuracy: ', accuracy4)
  print('Confusion Matrix: ', matrix4)
  print('Total Wrong Predictions: ', wrong_count4)