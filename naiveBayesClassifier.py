# https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
# https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import tarfile
import os
import glob
import shutil
import sys
import copy


class NaiveBayesClassifier():
    # Stores all the words regardless of any category
    total_words = dict()
    # Stores all the categories in the training document
    categories = dict()
    # Stores the words nested based on the new category it belongs to
    words_category = dict()
    # Stores the probabilities of each cat.
    category_probability = dict()
    # Stores the probabilities of each vocab word in cat.
    word_probability = dict()
    test_output = dict()

    def __init__(self, training_data):
        print("Training using the training data...")
        #Read data from the training file and put inside a dictionary
        with open(training_data, 'r') as fd:
            for line in fd.readlines():
                category, *words_category = line.split()
                self.categories[category] = self.categories.get(category, 0) + 1
                for word in words_category:
                    self.total_words[word] = self.total_words.get(word, 0) + 1
                    if category not in self.words_category.keys():
                        self.words_category[category] = {word: 1}
                    else:
                        self.words_category[category].update({word: self.words_category[category].get(word, 0) + 1})
        fd.close()

    def testing(self, testing_data):
        print("Testing the data using the trained data...")
        self.word_probability = copy.deepcopy(self.words_category)
        self.category_probability = copy.deepcopy(self.categories)
        for category in self.categories:
            self.category_probability[category] = self.categories.get(category, 1) / sum(self.categories.values())
            for word in self.words_category[category]:
                # Calculate probability of word in that category
                num_word_in_cat = self.words_category[category][word]
                num_tot_words_in_cat = sum(self.words_category[category].values())
                # print("P("+word+"|"+category+") = "+str(num_word_in_cat)+"/"+str(num_tot_words_in_cat))

                self.word_probability[category][word] = num_word_in_cat / num_tot_words_in_cat
        print("Finding the probabilities...")
        # print("Probability of words: "+str(self.word_probability))
        # print("Probability of categories: "+str(self.category_probability))

        # Dictionary to store the results
        results = dict()
        correct = 0
        total = 0

        # Testing the data
        with open(testing_data, 'r') as fd:
            for line in fd.readlines():
                total += 1  # Increment counter by 1
                result_category, *words_category = line.split()  # split testing into category and words
                # Iterate through categories of vocabulary
                for category in self.categories:
                    probability_category = self.category_probability[category]
                    probability_category_word = probability_category
                    for word in words_category:
                        probability_category_word *= self.word_probability[category].get(word, 0)
                    # print("P("+category+"|"+word+") = "+str(prob_category_word))
                    results.update({category: probability_category_word})
                    #Maximum of the probabilities is to be taken as the result
                result = max(results, key=results.get)
                self.get_output(result, result_category)
        #Find the total accuracy
        acc_total = 0
        print("\nThe results for each categories are as follows:\n")

        # print('{:30s} {:15s} {:15s} {:10s} '.format("category", "n_correct", "n_tot", "Percent"))
        for category in self.test_output:
            n_correct, n_tot = self.test_output[category]
            # print('{:29s} {:2f} {:2f} {:2f} '.format(category, n_correct, n_tot,n_correct/n_tot))
            percent = (n_correct / n_tot)
            acc_total = acc_total + percent
            print("\nCategory:  "+str(category))
            print("Accuracy:   "+str(percent))
            # print("\n")
            # print("\n\n")
            # print("Total accuracy obtained is:")
            # print(acc_total / 20)

    def get_output(self,result,r_category):
        if result == r_category and not(r_category in self.test_output):
            self.test_output.update({r_category:[1,1]})
        elif result != r_category and not(r_category in self.test_output):
            self.test_output.update({r_category:[0,1]})
        elif result == r_category:
            self.test_output.update({r_category:[ self.test_output[r_category][0]+1 , self.test_output[r_category][1]+1 ]})
        else:
            self.test_output.update({r_category:[ self.test_output[r_category][0] , self.test_output[r_category][1]+1 ]})


if __name__ == '__main__':
    fname = "../project2/input/20_newsgroups.tar.gz"
    classes = dict()

    train_data = []
    test_data = []

    #To separate train and test files and read
    # with os.scandir('../project2/train_files/') as entries:
    #     for entry in entries:
    #         with os.scandir('../project2/train_files/'+entry.name) as entry_name:
    #             for file in entry_name:
    #                 print(file.name)
    #                 f = open(file, 'r')
    #                 file_content = f.readlines()
    #                 train_data = train_data + file_content
    #                 f.close()
    #
    # s = ''
    # print(train_data)
    # train_text = s.join(train_data)
    # f = open("../project2/training_data.txt", "w+")
    # f.write(train_text)
    # print("training data put in txt file")

    # with os.scandir('../project2/test_files/') as entries:
    #     for entry in entries:
    #         with os.scandir('../project2/test_files/' + entry.name) as entry_name:
    #             for file in entry_name:
    #                 print(file.name)
    #                 f = open(file, 'r')
    #                 file_content = f.readlines()
    #                 test_data = test_data + file_content
    #                 f.close()
    #
    # s = ''
    # print(test_data)
    # test_text = s.join(test_data)
    # f = open("../project2/test_data.txt", "w+")
    # f.write(test_text)
    # print("testing data put in txt file")



    # Common stop words from online
    # stop_words = [
    #     "a", "about", "above", "across", "after", "afterwards",
    #     "again", "all", "almost", "alone", "along", "already", "also",
    #     "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow",
    #     "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become", "becomes",
    #     "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by",
    #     "can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else",
    #     "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find",
    #     "for", "found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence",
    #     "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how",
    #     "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made",
    #     "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my",
    #     "myself", "name", "namely", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor",
    #     "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other",
    #     "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "perhaps", "please", "put",
    #     "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should", "since", "sincere", "so",
    #     "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take",
    #     "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
    #     "therefore", "therein", "thereupon", "these", "they",
    #     "this", "those", "though", "through", "throughout",
    #     "thru", "thus", "to", "together", "too", "toward", "towards",
    #     "under", "until", "up", "upon", "us",
    #     "very", "was", "we", "well", "were", "what", "whatever", "when",
    #     "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    #     "wherein", "whereupon", "wherever", "whether", "which", "while",
    #     "who", "whoever", "whom", "whose", "why", "will", "with",
    #     "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
    # ]



    naiveBayes = NaiveBayesClassifier("../project2/train_files_preprocessed.txt")
    naiveBayes.testing("../project2/test_files_preprocessed.txt")
