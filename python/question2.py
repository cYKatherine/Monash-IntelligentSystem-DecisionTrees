import random
import math

def read_datafile(fname, attribute_data_type = 'integer'):
   inf = open(fname,'r')
   lines = inf.readlines()
   inf.close()
   #--
   X = []
   Y = []
   for l in lines:
      ss=l.strip().split(',')
      temp = []
      for s in ss:
         if attribute_data_type == 'integer':
            temp.append(int(s))
         elif attribute_data_type == 'string':
            temp.append(s)
         else:
            print("Unknown data type");
            exit();
      X.append(temp[:-1])
      Y.append(int(temp[-1]))
   return X, Y

#===
class Example:
   def __init__(self, features, outcome):
      self.features = features
      self.outcome = outcome


class DecisionTree :
   def __init__(self, split_random, depth_limit, curr_depth = 0, default_label = 1):
      self.split_random = split_random # if True splits randomly, otherwise splits based on information gain
      self.depth_limit = depth_limit
      self.parent = None
      self.default_label = default_label
      self.current_depth = 0

   def decision_tree_learning(self, examples, features_to_choose):
      if self.current_depth >= self.depth_limit:
         return self.plurality_value(examples)

      if len(examples) == 0:
         return self.default_label
      elif self.check_same_classification(examples):
         return examples[0].outcome
      elif not features_to_choose:
         return self.plurality_value(examples)
      else:
         if self.split_random:
            next_feature = random.choice(features_to_choose)  # Choose a random feature
         else:
            next_feature = # TODO: Choice feature based on entropy

   def get_entropy(self, examples):
      """ Return the value of entropy for the given examples. """
      if len(examples == 0):
         return 0

      1_count = 0
      0_count = 0
      for example in examples:
         if example.outcome == 1:
            1_count += 1
         else:
            0_count += 1

      positive = 1_count/len(examples)
      negative = 0_count/len(examples)

      if positive == 0 or negative == 0:
         return 0

      return -1 * positive * math.log(positive, 2) - negative * math.log(negative, 2)

   def check_same_classification(self, examples):
      """ Check if all the samples have the same outcome. """
      current = examples[0].outcome
      for i in range(1, len(examples)):
         if examples[i].outcome == current:
            continue
         else:
            return False
      return True

   def plurality_value(self, examples):
      """
      Return the most common output value among examples.
      When there is a tie, return randomly.
      """
      1_count = 0
      0_count = 0
      result = 0
      for i in range(len(examples)):
         if examples[i].outcome == 1:
            1_count += 1
         else:
            0_count += 1
      if 1_count > 0_count:
         result = 1
      elif 1_count == 0_count:
         result = random.choice([0, 1])
      return result

   def train(self, X_train, Y_train):
      # receives a list of objects of type Example
      # TODO: implement decision tree training
      pass

   def predict(self, X_train):
      # receives a list of booleans
      # TODO: implement decision tree prediction
      print("Kaaaaat")

#===
def compute_accuracy(dt_classifier, X_test, Y_test):
   numRight = 0
   for i in range(len(Y_test)):
      x = X_test[i]
      y = Y_test[i]
      if y == dt_classifier.predict(x) :
         numRight += 1
   return (numRight*1.0)/len(Y_test)

#==============================================
#==============================================
X_train, Y_train = read_datafile('train.txt')
X_test, Y_test = read_datafile('test.txt')
# TODO: write your code
