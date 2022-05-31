import pickle
import numpy as np
from collections import defaultdict

    
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def statistics_horizontal(word_dict, filename='5argStoriesCombined.txt'):
    positionDict = [dict() for x in range(0,5)]
    for j in range(0,5):
        for i,word in enumerate(word_dict[j]):
            positionDict[j][word] = i
    positionStats = [None]*5
    secondPositionStats = [None]*5

    for i in range(1, 5):
        positionStats[i] = np.zeros(shape = (len(positionDict[i]),len(positionDict[i-1])))
    for i in range(2, 3):
        secondPositionStats[i] = np.zeros(shape = (len(positionDict[i]),len(positionDict[i-2])))

    prev_word = None
    second_prev_word = None
    
    i = 0
    with open(filename,'r') as input:
        for line in input: 
            if "start_of_story" in line or "end_of_story" in line:
                continue
            else:
                words = line.strip().split(' ')
                for index, word in enumerate(words):
                    if index>0:
                        positionStats[index][positionDict[index][words[index]]][positionDict[index-1][words[index-1]]] += 1

                    if index==2:
                        secondPositionStats[index][positionDict[index][words[index]]][positionDict[index-2][words[index-2]]] += 1
            
    for i in range(1, 5):
        positionStats[i] = positionStats[i]/np.sum(positionStats[i])
    for i in range(2, 3):
        secondPositionStats[i] = secondPositionStats[i]/np.sum(secondPositionStats[i])
    #print("positionStats", positionStats)
    #print("secondPositionStats",secondPositionStats)
    return positionStats, secondPositionStats

def statistics(word_dict,filename='5argStoriesCombined.txt'):
    positionDict = [dict() for x in range(0,5)]
    for j in range(0,5):
        for i,word in enumerate(word_dict[j]):
            positionDict[j][word] = i
    positionStats = [None]*5
    secondPositionStats = [None]*5
    thirdPositionStats = [None]*5
    individualPositionStats = [None]*5

    for i in range(0,5):
        positionStats[i] = np.zeros(shape = (len(positionDict[i]),len(positionDict[i])))
    for i in range(0,5):
        secondPositionStats[i] = np.zeros(shape = (len(positionDict[i]),len(positionDict[i])))
    for i in range(0,5):
        thirdPositionStats[i] = np.zeros(shape = (len(positionDict[i]),len(positionDict[i])))
    for i in range(0,5):
        individualPositionStats[i] = np.zeros(shape = (len(positionDict[i])))

    prev_line = [None]*5
    second_prev_line = [None]*5
    third_prev_line = [None]*5
    i = 0
    # for filename in os.listdir(folder):
    with open(filename,'r') as input:
        for line in input:
            if "start_of_story" in line or "end_of_story" in line:
                i=0
                continue
            words = line.strip().split(' ')
            if i==0:                
                i+=1
                prev_line = words
            elif i==1:
                for j in range(0,5):
                    print(words[j])#, positionDict[j], prev_line[j])
                    positionStats[j][positionDict[j][words[j]]][positionDict[j][prev_line[j]]] += 1
                for j in range(0,5):
                    individualPositionStats[j][positionDict[j][words[j]]] += 1
                i+=1
                second_prev_line = prev_line
                prev_line = words
            elif i==2:
                for j in range(0,5):
                    positionStats[j][positionDict[j][words[j]]][positionDict[j][prev_line[j]]] += 1
                for j in range(0,5):
                    secondPositionStats[j][positionDict[j][words[j]]][positionDict[j][second_prev_line[j]]] += 1
                for j in range(0,5):
                    individualPositionStats[j][positionDict[j][words[j]]] += 1
                i+=1
                third_prev_line = second_prev_line
                second_prev_line = prev_line
                prev_line = words
            else:
                for j in range(0,5):
                    print("WORDS",words[j])
                    print("POS_DICT",positionDict[j])
                    print("PREV",prev_line[j])
                    print("POS_STAT",positionStats[j])
                    print("POS in POS_DICT",positionDict[j][words[j]])
                    print("POS in POS_DICT2",positionDict[j][prev_line[j]])
                    positionStats[j][positionDict[j][words[j]]][positionDict[j][prev_line[j]]] += 1
                for j in range(0,5):
                    secondPositionStats[j][positionDict[j][words[j]]][positionDict[j][second_prev_line[j]]] += 1
                for j in range(0,5):
                    thirdPositionStats[j][positionDict[j][words[j]]][positionDict[j][third_prev_line[j]]] += 1
                for j in range(0,5):
                    individualPositionStats[j][positionDict[j][words[j]]] += 1
                i+=1
                third_prev_line = second_prev_line
                second_prev_line = prev_line
                prev_line = words
    for i in range(0,5):
        positionStats[i] = positionStats[i]/np.sum(positionStats[i])
    for i in range(0,5):
        secondPositionStats[i] = secondPositionStats[i]/np.sum(secondPositionStats[i])
    for i in range(0,5):
        thirdPositionStats[i] = thirdPositionStats[i]/np.sum(thirdPositionStats[i])
    for i in range(0,5):
        individualPositionStats[i] = individualPositionStats[i]/np.sum(individualPositionStats[i])
    #print(np.amax(positionStats[0]))
    #print (individualPositionStats[0])
    #print(np.sum(individualPositionStats[0]))
    return positionStats,individualPositionStats,secondPositionStats,thirdPositionStats,positionDict

def compute_bigrams(in_file,out_file):
    #TODO: should I change this to include the 5th argument?
    word_dict2 = defaultdict(list)
    for position in range(0,5):
        # print "For position "+str(position)+" the number of words are:"
        with open(in_file,'r') as input:
            # putting all the words in a list before we count them
            for line in input:
                if "start_of_story" in line or "end_of_story" in line:
                    continue
                words=line.strip().split(',')
                if position==0 or position==2:
                    bi_gram = (words[position],words[position+1])
                    if bi_gram not in word_dict2[position]:
                        word_dict2[position].append(bi_gram)

    with open('words_for_sampling_bigrams' + '.pkl', 'wb') as f:
        pickle.dump(word_dict2, f, pickle.HIGHEST_PROTOCOL)

def noMoreEmptyParameter(filenameSource, filenameDest):
    emptyDict = {0: 'EmptySubject', 1: 'EmptyVerb', 2: 'EmptyObject', 3: 'EmptyModifier', 4: 'EmptyPreposition'}
    fout = open(filenameDest, 'w')
    with open(filenameSource,'r') as input:
        for line in input:
            if "start_of_story" in line or "end_of_story" in line:
                fout.write(line)
                continue
            words=line.strip().split(' ')
            for index, word in enumerate(words):
                if word == "EmptyParameter" or word == 'emptyparameter':
                    fout.write(emptyDict[index])
                else:
                    fout.write(word)
                if index != 4:
                    fout.write(',')
                else:
                    fout.write('\n')
    fout.close()



