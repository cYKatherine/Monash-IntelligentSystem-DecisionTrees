import argparse as ap
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
      self.default_label = default_label
      self.current_depth = 0

      self.parent = None
      self.subset_0 = None
      self.subset_1 = None

      self.chosen_feature = None

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
            next_feature = self.choose_feature_on_entropy(examples, features_to_choose)

         self.chosen_feature = next_feature

         # TODO: Now we have the new subsets, HOW TO LINK THE NEW SUBSET TO THE CURRENT TREE?????
         new_examples_0, new_examples_1 = self.get_new_examples(examples, next_feature)
         new_features_to_choose = features_to_choose[:]
         new_features_to_choose.remove(next_feature)
         new_default = self.plurality_value(examples)

         next_examples = [new_examples_0, new_examples_1]
         for i in range(1):
            next_example = next_examples[i]
            subtree = DecisionTree(
               self.split_random, self.depth_limit, self.current_depth + 1, new_default
            ).decision_tree_learning(next_example, new_features_to_choose)

            if i == 0:
               self.subset_0 = subtree
            else:
               self.subset_1 = subtree

            if subtree not in [0, 1]:
               subtree.parent = self

         return self

   def choose_feature_on_entropy(self, examples, features_to_choose):
      """ Return the next feature to choose based on the optimal entropy. """
      current_entropy = get_entropy(examples)
      max_information_gain = -1
      next_feature_index = -1

      for feature_index in features_to_choose:
         new_examples_0, new_examples_1 = self.get_new_examples(examples, feature_index)

         # Get the entropy of subsets
         new_examples_0_entropy = self.get_entropy(new_examples_0)
         new_examples_1_entropy = self.get_entropy(new_examples_1)

         new_examples_0_length = len(new_examples_0)
         new_examples_1_length = len(new_examples_1)
         current_examples_length = len(examples)
         subset_entropy_expectation = (new_examples_0_length / current_examples_length) * new_examples_0_entropy + (new_examples_1_length / current_examples_length) * new_examples_1_entropy

         information_gain = current_entropy - subset_entropy_expectation

         if information_gain > max_information_gain:
            max_information_gain = information_gain
            next_feature_index = feature_index

      return next_feature_index

   def get_new_examples(self, examples, feature_index):
      """ Split the current exmaples based on `feature_index`. """
      new_examples_0 = []
      new_examples_1 = []
      for row in examples:
         new_row = row[:feature_index] + row[feature_index+1:]
         if row[feature_index] == 0:
            new_examples_0.append(new_row)
         else:
            new_examples_1.append(new_row)
      return (new_examples_0, new_examples_1)

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
      examples = []
      for i in range(len(X_train)):
         example = Example(X_train[i], Y_train[i])
         examples.append(example)

      features_to_choose = []
      for i in range(len(X_train[0])):
         features_to_choose.append(i)

      return self.decision_tree_learning(examples, features_to_choose)

   def predict(self, X_test):
      # receives a list of booleans
      # TODO: implement decision tree prediction
      res = []
      for test_case in X_test:
         res.append(self.predict_aux(test_case))

   def predict_aux(self, test_case):
      check_data = test_case[self.chosen_feature]
      if check_data == 0:
         if self.subset_0 in [0, 1]:
            return self.subset_0
         else:
            self.subset_0.predict_aux(test_case)
      elif check_data == 1:
         if self.subset_1 in [0, 1]:
            return self.subset_1
         else:
            self.subset_1.predict_aux(test_case)

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
def write_to_file(file_name, solution):
    file_handle = open(file_name, 'w')
    file_handle.write(solution)

def main():
    # create a parser object
    parser = ap.ArgumentParser()

    # specify what arguments will be coming from the terminal/commandline
    parser.add_argument("train_file", help= "the  name  of  the  input  training  file", type= str)
    parser.add_argument("spliting_method", help="he  splitting  method  which  is  either  Information  gain  or  Random  (denoted  by  I  or  R  respectively)", type=str)
    parser.add_argument("depth", help="the  maximum  depth  allowed  for  the  decision  tree,", type=int)
    parser.add_argument("test_file", help="the name of the test file", type=str)
    parser.add_argument("output_file", help="the name of the output  file into which the accuracy of the learned model on the test set is written into a  line", type=str)

    # get all the arguments
    arguments = parser.parse_args()

    X_train, Y_train = read_datafile(arguments.train_file)
    X_test, Y_test = read_datafile(arguments.test_file)
    # TODO: write your code
    tree = DecisionTree(True if arguments.spliting_method == 'R' else False, arguments.depth, 0, 1)
    # import pdb; pdb.set_trace()
    tree.train(X_train, Y_train)
    print(compute_accuracy(tree, X_test, Y_test))
#==============================================
if __name__ == "__main__":
    main()
