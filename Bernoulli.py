import math
import numpy as np

class Bernoulli:

    __model = {}
    __prob_of_Spam = 0.16  # prior probability of spam
    __prob_of_Ham = 0.84  # prior probability of ham

    def Train(self, SPAM_set, HAM_set): # calculate number of words
        SPAM = {}
        HAM = {}

        # ------------SPAM-------------
        for i in range(len(SPAM_set)):
            if SPAM_set[i] in SPAM:  # incrementing counter if word in spam dict
                SPAM[SPAM_set[i]] += 1
            else:
                SPAM[SPAM_set[i]] = 1  # add new word to dictionary
        for i in SPAM:
            SPAM[i] = (SPAM[i]+1)/(len(SPAM)+2)  # calculate probabilities

        # ------------HAM--------------
        for i in range(len(HAM_set)):
            if HAM_set[i] in HAM:  # incrementing counter if word in ham dict
                HAM[HAM_set[i]] += 1
            else:
                HAM[HAM_set[i]] = 1  # add new word to dictionary
        for i in HAM:
            HAM[i] = (HAM[i]+1)/(len(HAM)+2)  # calculate probabilities
        # ---------------------------------

        self.__model["SPAM"] = SPAM
        self.__model["HAM"] = HAM
        self.__SPAM_count = len(SPAM_set)  # set spam counter to length of spam set
        self.__HAM_count = len(HAM_set)  # set ham counter to length of ham set


    def Predict(self, predict_set):  # predict probability
        Prob = []
        spam_Prob = []
        for i in range(len(predict_set)):
            prob = math.log(self.__prob_of_Spam)
            for j in range(len(predict_set[i])):
                if predict_set[i][j] in self.__model["SPAM"]:
                    # calculate probability of spam
                    prob += math.log(self.__model["SPAM"][predict_set[i][j]])
                else:
                    prob += math.log(1/(self.__SPAM_count+2))  # if word not in dictionary
            for k in self.__model["SPAM"]:  # calculating probability that word not from SPAM
                if k not in predict_set[i]:
                    prob += math.log(1-self.__model["SPAM"][k])
            spam_Prob.append(prob)

        ham_Prob = []
        for i in range(len(predict_set)):
            prob = math.log(self.__prob_of_Ham)
            for j in range(len(predict_set[i])):
                if predict_set[i][j] in self.__model["HAM"]:
                    # calculate probability of ham
                    prob += math.log(self.__model["HAM"][predict_set[i][j]])
                else:
                    prob += math.log(1 / (self.__HAM_count + 2))   # if word not in dictionary
            for k in self.__model["HAM"]:  # calculating probability that word not from HAM
                if k not in predict_set[i]:
                    prob += math.log(1 - self.__model["HAM"][k])
            ham_Prob.append(prob)

        for i in range(len(spam_Prob)):
            Prob.append(max(spam_Prob[i]-ham_Prob[i], 0))  # probability of spam

        return Prob

