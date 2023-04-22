# # TextClassifier.py
# # ---------------
# # Licensing Information:  You are free to use or extend this projects for
# # educational purposes provided that (1) you do not distribute or publish
# # solutions, (2) you retain this notice, and (3) you provide clear
# # attribution to the University of Illinois at Urbana-Champaign
# #
# # Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019
"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


from queue import PriorityQueue
import math

class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0 # lambda list for mixed uni-N-gram
        self.N = 1 # number of gram, 1 for unigram, 2 for bigram, n for Ngram
        self.prior_distribution = 1 # 1 log_prior , 2 uniform, else not included
        self.ngrams_in_class = {} # \lambda_cg, {c:{gram:count_g}}
        self.total_ngrams_in_class={}# \lambda, {c:count_g}
        
        self.classes={} # set of classes {c:count_c}
        self.log_prior = {} # \pai_c, {c:log_p(c)}
        
        self.words_in_class={}# \lambda_cw, {c:{w:count_w}}
        self.total_words_in_class={}# \lambda, {c:count_w}
        
        self.confusion_matrix = {} # {r:{c:p=confusion_r_to_c/count_r}}
        self.v = set() # {word}
        self.v_gram = set() # {gram}
        
        self.log_file="log.txt"
        self.f= open(self.log_file,"w").close()

        # MNB.calculate_cm(x_test, y_test)
        # MNB.print_cm()
        # MNB.top_20_word()
        # MNB.top_20_ngram()
        # MNB.find_best_lambda(x_test, y_test)
        # MNB.find_best_N(x_test, y_test)  
        
    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """
    
        train_data = list(zip(train_set,train_label))
        for words,label in train_data:
            # self.logging(str([words,label]))
            if label not in self.classes.keys():
                self.classes[label] = 0
                self.words_in_class[label] = {}
                self.total_words_in_class[label] = 0
                self.ngrams_in_class[label] = {}
                self.total_ngrams_in_class[label] = 0
            self.classes[label] += 1
            
            for word in words:
                if word not in self.words_in_class[label].keys():
                    self.words_in_class[label][word] = 0
                
                self.words_in_class[label][word] += 1
                self.total_words_in_class[label] += 1

                if word not in self.v:
                    self.v.add(word)
                    
            #fit n grams
            for i in range(len(words)-self.N+1):
                ngram = tuple(words[i:i+self.N])
                # self.logging("\nngram : {}".format(ngram))

                if ngram not in self.ngrams_in_class[label].keys():
                    self.ngrams_in_class[label][ngram] = 0

                self.ngrams_in_class[label][ngram] += 1
                self.total_ngrams_in_class[label] += 1

                if ngram not in self.v_gram:
                    self.v_gram.add(ngram)

        num_doc = len(train_data)
  
        for c in self.classes.keys():
            self.log_prior[c] = math.log(self.classes[c]) - math.log(num_doc)
        
        
        

    def predict(self, x_set, dev_label,lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """
        self.lambda_mixture = lambda_mix
        accuracy = 0.0
        result = []
        test_data = list(zip(x_set, dev_label))
        length = len(test_data)
        for words,label in test_data:
            y_pred =self.mixed_Ngram_predict(words)
            result.append(y_pred)
            if y_pred == label:
                accuracy += 1
   	
        accuracy = accuracy/length
        
        return accuracy,result

    def mixed_Ngram_predict(self,words):
        pq = PriorityQueue()

        for c in self.classes.keys():
            log_P_class = 0
            if self.prior_distribution == 1:
                log_P_class = self.log_prior[c]
            elif self.prior_distribution == 2:
                log_P_class = math.log(1 / len(self.classes))

            # Compute the probability of each words in class c
            log_P_word_class = 0
            for w in words:
                if w in self.words_in_class[c].keys():
                    log_P_word_class += math.log(self.words_in_class[c][w]+ 1) - math.log(self.total_words_in_class[c]+len(self.v))
                else:
                    log_P_word_class += - math.log(self.total_words_in_class[c]+len(self.v))
            
            # Compute the probability of each n-gram in class c
            log_P_ngram_class = 0
            # if self.N != 1:
            for i in range(len(words)-self.N+1):
                ngram = tuple(words[i:i+self.N])
                # print(ngram)
                # self.logging("\nngram : {}".format(ngram))
                if ngram in self.ngrams_in_class[c].keys():
                    log_P_ngram_class += math.log(self.ngrams_in_class[c][ngram]+ 1) - math.log(self.total_ngrams_in_class[c]+len(self.v_gram))
                else:
                    log_P_ngram_class += - math.log(self.total_ngrams_in_class[c]+len(self.v_gram))
            # print(log_P_ngram_class-log_P_word_class)
            # Combine the probabilities of the n-gram and the words in the n-gram
            log_P = log_P_class + (1 - self.lambda_mixture) * log_P_word_class + self.lambda_mixture *  log_P_ngram_class

            pq.put([1 - log_P, c]) # sort by min, so max log_p will pop first

        y_pred = pq.get()[1]

        return y_pred

    def calculate_cm(self, x_set, dev_label):
        test_data = list(zip(x_set, dev_label))
        for words,r in test_data:
            c =self.mixed_Ngram_predict(words)
            # if c != r:
            if r not in self.confusion_matrix.keys():
                self.confusion_matrix[r]={}
            elif c not in self.confusion_matrix[r].keys():
                self.confusion_matrix[r][c] = 1.0
            else:
                self.confusion_matrix[r][c] += 1.0

        for r in self.confusion_matrix.keys():
            total_r = sum(self.confusion_matrix[r].values())
            for c in self.confusion_matrix[r].keys():
                self.confusion_matrix[r][c] /= total_r

        return self.confusion_matrix

    def print_cm(self):
        self.logging("\nconfusion_matrix : ")
        with open(self.log_file,"a") as f:
            # f.write("confusion_matrix: \n")
            for i in range(len(self.classes)):
                f.write("|")
                for j in range(len(self.classes)):
                    r = i+1
                    c = j+1
                    if c not in self.confusion_matrix[r].keys() or r not in self.confusion_matrix.keys():
                        f.write(" {:.3f} |".format(0.0))
                    else:
                        f.write(" {:.3f} |".format(self.confusion_matrix[r][c]))
                f.write("\n")    
            f.write("\n")    

    def top_20_word(self):
        self.logging("\ntop_20_word : ")
        for i in range(len(self.classes)):
            c=i+1
            sorted_words=sorted(self.words_in_class[c].items(), key = lambda kv:kv[1], reverse=True)[:20]
            self.logging("{} : {}".format(c,sorted_words))
    
    def top_20_ngram(self):
        self.logging("\ntop_20_ngram : ")
        for i in range(len(self.classes)):
            c=i+1
            sorted_ngrams=sorted(self.ngrams_in_class[c].items(), key = lambda kv:kv[1], reverse=True)[:20]
            self.logging("{} : {}".format(c,sorted_ngrams))  
        
    def logging(self,log:str):
        with open(self.log_file,"a") as f:
            print(log,file=f)
    
    def clear(self):
        self.ngrams_in_class = {} # \lambda_cg, {c:{gram:count_g}}
        self.total_ngrams_in_class={}# \lambda, {c:count_g}
        
        self.classes={} # set of classes {c:count_c}
        self.log_prior = {} # \pai_c, {c:log_p(c)}
        
        self.words_in_class={}# \lambda_cw, {c:{w:count_w}}
        self.total_words_in_class={}# \lambda, {c:count_w}
        
        self.confusion_matrix = {} # {r:{c:p=confusion_r_to_c/count_r}}
        self.v = set() # {word}
        self.v_gram = set() # {gram}
        
    def find_best_lambda(self,x_train, y_train, x_set, dev_label):
        self.N=2
        self.clear()
        self.fit(x_train, y_train)
        self.logging("\n\n\nfind_best_lambda : ")
        lambda_mixture = 0.0
        step=1000
        accuracy_pq=PriorityQueue()
        for i in range(0,step+1):
            lambda_mixture = i/step
            accuracy,pred = self.predict(x_set, dev_label,lambda_mixture)
            self.logging("lambda : {}, {}".format(lambda_mixture,accuracy))
            accuracy_pq.put((1-accuracy,lambda_mixture))

            
        
        best_accuracy,best_lambda_mixture = accuracy_pq.get()
        self.logging("\n\nbest_lambda_mixture is : {}, {}".format(best_lambda_mixture,1-best_accuracy))
        return
    
    def find_best_N(self, x_train, y_train, x_set, dev_label):
        self.logging("\n\n\nfind_best_lambda : ")
        lambda_mixture = 1.0
        step=1000
        accuracy_pq=PriorityQueue()
        for i in range(1,step+1):
            self.N=i
            self.clear()
            self.fit(x_train, y_train)
            accuracy,pred = self.predict(x_set, dev_label,lambda_mixture)
            self.logging("self.N: {},{}".format(self.N,accuracy))
            accuracy_pq.put((1-accuracy,self.N))
        
        best_accuracy,best_N = accuracy_pq.get()
        self.logging("\n\nbest_N is : {}, {}".format(best_N,1-best_accuracy))
        return