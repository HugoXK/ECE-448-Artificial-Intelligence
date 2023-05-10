# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    data = {}
    count = {}
    tags = []
    # traverse all sentence
    for sentence in train:
        # traverse all word
        for word in sentence:
            # initilization: if not found before, then add
            if data.get(word[1]) is None:
                data[word[1]] = {}
                count[word[1]] = 0
                tags.append(word[1])
            count[word[1]] += 1
            if data[word[1]].get(word[0]) is None:
                data[word[1]][word[0]] = 1
            else:
                data[word[1]][word[0]] += 1
    # print(f'-------tags------{tags}')
    # print(f'-------count------{count}')
    # print(f'-------data------{data}')

    # initilization: begin with the one occur most
    common = max(count, key=count.get)
    predicts = []
    for i in range(len(test)):
        # initilization: empty but one to one map space
        predicts.append([])
        for word in test[i]:
            predict = common
            # initilization: as the min occurance in count dict is 0
            counter = -1
            for tag in tags:
                if data[tag].get(word) is not None:
                    if data[tag].get(word) > counter:
                        # replace if more likely found
                        counter = data[tag].get(word)
                        predict = tag
            predicts[i].append((word, predict))
    #print(f'-------predicts------{predicts}')

    return predicts