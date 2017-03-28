import nltk
from nltk.corpus import brown
import random

'''
  Myanna Harris
  Jasmine Jans (submitter)
  10-26-16
  asgn5.py

  N-Gram generator:
  This program uses the Brown corpus to calculate frequencies and prababilities of
  certain work sequences occuring, and from that creates 4 functions to generate
  uni, bi, tri and quadgrams.

  Note: Read comment in name regarding size of the corpus. We had to reduce its
  size to get the program to run at a resonable speed.
  Also, we could not elimintate the '' that appeared in the corpus, so it sometimes
  appears as a word.
  
  To run:
  python asgn5.py 
'''

'''
prepares the corpus by removing unicode, and ading a <s> to the begining
of the sentence and replacing end punctuation with a </s>
'''
def preparingCorpus(lists):
    newLists = [[] for i in range(0,len(lists))]

    wordIdxDict = {}
    currDictIdx = 0

    wordCountTotal = 0

    #go through all the sentences in the corpus and rebuild the lists
    for i in range(0,len(lists)):
        #add the <s> token to the begining
        newLists[i].append("<s>")
        wordCountTotal += 1
        if not wordIdxDict.has_key("<s>"):
            wordIdxDict["<s>"] = currDictIdx
            currDictIdx += 1

        #verifies words are out of unicode
        for k in range(0, len(lists[i])):
            asciiWord = lists[i][k].encode('ascii', 'ignore')
            if not (k > len(lists[i])-3 and
                (asciiWord == '.' or asciiWord == '?'
                  or asciiWord == '!' or asciiWord == ':'
                  or asciiWord == ';' or asciiWord == "''")):
                
                newLists[i].append(asciiWord)

                #adds to word total, and creates index for new word
                wordCountTotal += 1
                if not wordIdxDict.has_key(asciiWord):
                    wordIdxDict[asciiWord] = currDictIdx
                    currDictIdx += 1

        #add the </s> token to the end
        newLists[i].append("</s>")
        wordCountTotal += 1
        if not wordIdxDict.has_key("</s>"):
            wordIdxDict["</s>"] = currDictIdx
            currDictIdx += 1

    return (newLists, wordIdxDict, wordCountTotal)

'''
Stores the unigrams and their respective probabilities in a
list and then returns that list
'''
def makeUnigramStructure(lists, wordIdxDict, wordCountTotal):
    unigramMatrix = [0 for i in range(0, len(wordIdxDict.keys()))]

    unigramFreq = {}

    for i in range(0,len(lists)):
        for k in range(0, len(lists[i])):
            word = lists[i][k]
            wordIdx = wordIdxDict[word]
            unigramMatrix[wordIdx] += 1

            #update the unigram frequency dictionary
            if not unigramFreq.has_key(word):
                unigramFreq[word] = 1
            else:
                unigramFreq[word] += 1
                
    #calculate the probabilities of the unigrams, and put in a list
    for i in range(0,len(unigramMatrix)):
        unigramMatrix[i] /= float(wordCountTotal)
            
    return (unigramMatrix, unigramFreq)

'''
Stores bigrams and their respective probabilities in a matrix
(unigramXunigram) and returns the matrix
'''
def makeBigramStructure(lists, wordIdxDict, unigrams, unigramFreq):
    bigramMatrix = [[0 for i in range(0, len(wordIdxDict.keys()))]
                    for k in range(0, len(wordIdxDict.keys()))]

    bigramIdxDict = {}
    currDictIdx = 0

    bigramCountTotal = 0

    bigramFreq = {}

    for i in range(0,len(lists)):
        for k in range(1, len(lists[i])):
            prevWord = lists[i][k-1]
            word = lists[i][k]
            prevWordIdx = wordIdxDict[prevWord]
            wordIdx = wordIdxDict[word]
            #add freq of the bigram to correlating matrices position
            bigramMatrix[wordIdx][prevWordIdx] += 1

            #adds a correlating bigram index
            if not bigramIdxDict.has_key(prevWord +" "+ word):
                bigramIdxDict[prevWord +" "+ word] = currDictIdx
                currDictIdx += 1

            #adds to the frequency of the bigram
            if not bigramFreq.has_key(prevWord +" "+ word):
                bigramFreq[prevWord +" "+ word] = 1
            else:
                bigramFreq[prevWord +" "+ word] += 1

    #calculate the conditional probability of each bigram
    for i in range(0,len(bigramMatrix)):
        for k in range(0, len(bigramMatrix[i])):
            unigramWords = ""
            for key, value in bigramIdxDict.items():
                if value == k:
                    unigramWords = key.split()
                    break
            unigramWord = unigramWords[0]
            bigramMatrix[i][k] /= float(unigramFreq[unigramWord])
            
    return (bigramMatrix, bigramIdxDict, bigramFreq)

'''
Stores trigrams and their respective probabilities in a matrix
(bigramXunigram) and returns the matrix
'''
def makeTrigramStructure(lists, wordIdxDict, bigrams, bigramIdxDict, bigramFreq):
    trigramMatrix = [[0 for i in range(0, len(bigramIdxDict.keys()))]
                    for k in range(0, len(wordIdxDict.keys()))]

    trigramIdxDict = {}
    currDictIdx = 0

    trigramFreq = {}

    for i in range(0,len(lists)):
        for k in range(2, len(lists[i])):
            prevWords = lists[i][k-2]+ " " + lists[i][k-1]
            word = lists[i][k]
            prevWordsIdx = bigramIdxDict[prevWords]
            wordIdx = wordIdxDict[word]
            #add freq of the trigram to correlating matrices position
            trigramMatrix[wordIdx][prevWordsIdx] += 1

            #adds a correlating trigram index
            if not trigramIdxDict.has_key(prevWords +" "+ word):
                trigramIdxDict[prevWords +" "+ word] = currDictIdx
                currDictIdx += 1

            #adds to the frequency of the trigram
            if not trigramFreq.has_key(prevWords +" "+ word):
                trigramFreq[prevWords +" "+ word] = 1
            else:
                trigramFreq[prevWords +" "+ word] += 1

    #calculate the conditional probability of each trigram
    for i in range(0,len(trigramMatrix)):
        for k in range(0, len(trigramMatrix[i])):
            bigramWords = ""
            for key, value in bigramIdxDict.items():
                if value == k:
                    bigramWords = key.split()
                    break
            bigram = bigramWords[0] + " " + bigramWords[1]
            trigramMatrix[i][k] /= float(bigramFreq[bigram])
            
    return (trigramMatrix, trigramIdxDict, trigramFreq)

'''
Stores quadgrams and their respective probabilities in a matrix
(trigramXunigram) and returns the matrix
'''
def makeQuadgramStructure(lists, wordIdxDict, trigrams, trigramIdxDict, trigramFreq):
    quadgramMatrix = [[0 for i in range(0, len(trigramIdxDict.keys()))]
                    for k in range(0, len(wordIdxDict.keys()))]

    for i in range(0,len(lists)):
        for k in range(3, len(lists[i])):
            prevWords = lists[i][k-3]+" " +lists[i][k-2]+" " +lists[i][k-1]
            word = lists[i][k]
            prevWordsIdx = trigramIdxDict[prevWords]
            wordIdx = wordIdxDict[word]
            #add freq of the quadgram to correlating matrices position
            quadgramMatrix[wordIdx][prevWordsIdx] += 1

    #calculate the conditional probability of each quadgram
    for i in range(0,len(quadgramMatrix)):
        for k in range(0, len(quadgramMatrix[i])):
            trigramWords = ""
            for key, value in trigramIdxDict.items():
                if value == k:
                    trigramWords = key.split()
                    break
            trigram = trigramWords[0] + " " + trigramWords[1] + " " + trigramWords[2]
            quadgramMatrix[i][k] /= float(trigramFreq[trigram])
            
    return quadgramMatrix

'''
Creates a list of continuous probabilities correlating to
the probabilities of the unigram list
'''
def continuousProbabilityUni(ngrams):
    continuousProbs = [0 for i in range(len(ngrams))]
    
    for i in range(len(continuousProbs)):
        if i == 0:
            continuousProbs[i] = ngrams[i]
        elif i == len(continuousProbs)-1:
            continuousProbs[i] = 1
        else:
            continuousProbs[i] = continuousProbs[i-1]+ngrams[i]

    return continuousProbs

'''
Creates a matrix of continuous probabilities correlating to
the probabilities of the n-gram list (n>1). It does this by
using the unigram method above, where each row in the matrix is a list
'''
def continuousProbability(ngrams):
    continuousProbs = [[0 for i in range(len(ngrams[0]))] for j in range(len(ngrams))]
    
    for i in range(len(continuousProbs[0])):
        for j in range(len(continuousProbs)):
            if j == 0:
                continuousProbs[j][i] = ngrams[j][i]
            elif j == len(continuousProbs)-1:
                continuousProbs[j][i] = 1
            else:
                continuousProbs[j][i] = continuousProbs[j-1][i] + ngrams[j][i]

    return continuousProbs

'''
randomly generates a sentence of unigrams that starts with <s>
and will complete with </s>
'''
def makeUnigramSentence(unigrams, wordIdxDict):
    continuousProbs = continuousProbabilityUni(unigrams)

    sentence = ""
    word = ""

    #generate random numbers till a <s> is found
    while word != "<s>":
        randomNum = random.random()
        index = 0

        for i in range(0, len(unigrams)):
            if randomNum <= continuousProbs[i]:
                index = i
                break

        for key, value in wordIdxDict.items():
            if value == index:
                word = key
                break

    sentence += word

    #generate random numbers and select correlating unigrams till a </s> is found
    while word != "</s>":
        randomNum = random.random()
        index = 0

        for i in range(0, len(unigrams)):
            if randomNum <= continuousProbs[i]:
                index = i
                break
        
        for key, value in wordIdxDict.items():
            if value == index and key != "<s>":
                word = key
                break
                    
        if (word != "<s>"):   
            sentence += " " + word
        else:
            word = ""
    
    return sentence

'''
randomly generates a sentence of bigrams that starts with <s>
and will complete with </s>
'''
def makeBigramSentence(bigrams, wordIdxDict):
    continuousProbs = continuousProbability(bigrams)

    sentence = ""
    word1 = ""
    word2 = ""

    #generate random numbers till a <s> is found as first word in bigram
    while word1 != "<s>":
        randomNum1 = random.randint(0,len(bigrams[0])-1)
        randomNum2 = random.random()
        index1 = 0
        index2 = 0

        for j in range(0, len(bigrams)):    
            if randomNum2 <= continuousProbs[j][randomNum1]:
                index1 = randomNum1
                index2 = j
                break

        #split the bigrams into 2 separate words
        for key, value in wordIdxDict.items():
            if value == index1:
                word1 = key
            if value == index2:
                word2 = key

    sentence += word1 + " " + word2

    #generate random numbers and select correlating bigrams till a </s> is found
    #as the second word of a bigram
    while word2 != "</s>":
        randomNum1 = random.randint(0,len(bigrams[0])-1)
        randomNum2 = random.random()
        index1 = 0
        index2 = 0

        for j in range(0, len(bigrams)):    
            if randomNum2 <= continuousProbs[j][randomNum1]:
                index1 = randomNum1
                index2 = j
                break

        for key, value in wordIdxDict.items():
            if value == index1 and key != "<s>":
                word1 = key
            if value == index2 and key != "<s>":
                word2 = key

        if (word1 != "<s>" and word2 != "<s>" and word1 != "</s>"):   
            sentence += " " + word1 + " " + word2
        else:
            word1 = ""
            word2 = ""
    
    return sentence
    
'''
randomly generates a sentence of trigrams that starts with <s>
and will complete with </s>
'''
def makeTrigramSentence(trigrams, bigramIdxDict, wordIdxDict):
    continuousProbs = continuousProbability(trigrams)

    sentence = ""
    word1 = ""
    word2 = ""
    word1b = ""

    #generate random numbers till a <s> is found as first word in trigram
    while word1 != "<s>":
        randomNum1 = random.randint(0,len(trigrams[0])-1)
        randomNum2 = random.random()
        index1 = 0
        index2 = 0

        for j in range(0, len(trigrams)):    
            if randomNum2 <= continuousProbs[j][randomNum1]:
                index1 = randomNum1
                index2 = j
                break

        for key, value in wordIdxDict.items():
            if value == index2:
                word2 = key
                break

        #split the trigrams into 3 separate words 
        for key, value in bigramIdxDict.items():
            key1, key2 = key.split()
            if value == index1:
                word1 = key1
                word1b = key2
                break

    sentence += word1 + " " + word1b + " " + word2

    #generate random numbers and select correlating trigrams till a </s> is found
    #as the third word of a trigram
    while word2 != "</s>" and word1 != "</s>" and word1b != "</s>":
        randomNum1 = random.randint(0,len(trigrams[0])-1)
        randomNum2 = random.random()
        index1 = 0
        index2 = 0

        for j in range(0, len(trigrams)):    
            if randomNum2 <= continuousProbs[j][randomNum1]:
                index1 = randomNum1
                index2 = j
                break

        #split the bigram used in the trigram
        for key, value in bigramIdxDict.items():
            key1, key2 = key.split()
            if value == index1 and key1 != "<s>" and key2 != "<s>" and key1 != "</s>" and key2 != "</s>":
                word1 = key1
                word1b = key2
                break

        #find the unigram used in the trigram
        for key, value in wordIdxDict.items():
            if value == index2 and key != "<s>":
                word2 = key
                break

        #contstruct the sentence  
        if (word1 != "<s>" and word1b != "<s>" and word2 != "<s>"
            and word1 != "</s>" and word1b != "</s>"):   
            sentence += " " + word1 + " " + word1b + " " + word2
        else:
            word1 = ""
            word1b = ""
            word2 = ""
        
    return sentence

'''
randomly generates a sentence of quadgrams that starts with <s>
and will complete with </s>
'''
def makeQuadgramSentence(quadgrams, trigramIdxDict, wordIdxDict):
    continuousProbs = continuousProbability(quadgrams)

    sentence = ""
    word1 = ""
    word2 = ""
    word1b = ""
    word1c = ""

    #generate random numbers till a <s> is found as first word in quadgram
    while word1 != "<s>":
        randomNum1 = random.randint(0,len(quadgrams[0])-1)
        randomNum2 = random.random()
        index1 = 0
        index2 = 0

        for j in range(0, len(quadgrams)):    
            if randomNum2 <= continuousProbs[j][randomNum1]:
                index1 = randomNum1
                index2 = j
                break

        for key, value in wordIdxDict.items():
            if value == index2:
                word2 = key
                break

        #split the quadgram into 4 separate words 
        for key, value in trigramIdxDict.items():
            key1, key2, key3 = key.split()
            if value == index1:
                word1 = key1
                word1b = key2
                word1c = key3
                break

    sentence += word1 + " " + word1b + " " + word1c + " " + word2

    #generate random numbers and select correlating trigrams till a </s> is found
    #as the fourth word of a quadgram
    while word2 != "</s>" and word1 != "</s>" and word1b != "</s>" and word1c != "</s>":
        randomNum1 = random.randint(0,len(quadgrams[0])-1)
        randomNum2 = random.random()
        index1 = 0
        index2 = 0

        for j in range(0, len(quadgrams)):    
            if randomNum2 <= continuousProbs[j][randomNum1]:
                index1 = randomNum1
                index2 = j
                break

        #split the trigram used in the quadgram
        for key, value in trigramIdxDict.items():
            key1, key2, key3 = key.split()
            if (value == index1 and key1 != "<s>" and key2 != "<s>"
                and key3 != "<s>" and key1 != "</s>" and key2 != "</s>" and key3 != "</s>"):
                word1 = key1
                word1b = key2
                word1c = key3
                break

        #find the unigram used in the quadgram
        for key, value in wordIdxDict.items():
            if value == index2 and key != "<s>":
                word2 = key
                break

        #contstruct the sentence 
        if (word1 != "<s>" and word1b != "<s>" and word1c != "<s>" and word2 != "<s>"
            and word1 != "</s>" and word1b != "</s>" and word1c != "</s>"):   
            sentence += " " + word1 + " " + word1b + " " + word1c + " " + word2
        else:
            word1 = ""
            word1b = ""
            word1c = ""
            word2 = ""
        
    return sentence

def main():
    '''
    calls to add sentence markers to sentences for news
    
    wordIdxDict =  dictionary telling the index for each word
                    for future use in N-gram structures
    wordCountTotal = number of total words in the corpus
    '''
    
    news = brown.sents(categories='editorial')
    new_news = [[item.encode('ascii') for item in lst] for lst in news]

    #the program runs slow (we have yet to experience it complete) when run on the whole corpus
    #we use this to shorten the corpus (takes about 5-10 minutes at 100 lines)
    news = [new_news[i] for i in range(0, 50)]
    news, wordIdxDict, wordCountTotal = preparingCorpus(news)

    #gets the unigrams, and unigram frequency
    unigrams, unigramFreq = makeUnigramStructure(news, wordIdxDict, wordCountTotal)

    #gets the bigrams, their respective indexes in the matrix, and the bigrams freqency
    bigrams, bigramIdxDict, bigramFreq = makeBigramStructure(news, wordIdxDict, unigrams, unigramFreq)

    #gets the trigrams, their respective indexes in the matrix, and the trigrams freqency
    trigrams, trigramIdxDict, trigramFreq = makeTrigramStructure(news, wordIdxDict, bigrams, bigramIdxDict, bigramFreq)

    #gets the quadgrams, their respective indexes in the matrix, and the quadgrams freqency 
    quadgrams = makeQuadgramStructure(news, wordIdxDict, trigrams, trigramIdxDict, trigramFreq)

    print "------------------------------------------------------"
    print "Unigrams"
    print "------------------------------------------------------"
    count = 0
    while(count<5):
        print(makeUnigramSentence(unigrams, wordIdxDict))
        print()
        count+=1

    print "------------------------------------------------------"
    print "Bigrams"
    print "------------------------------------------------------"
    count = 0
    while(count<5):
        print(makeBigramSentence(bigrams, wordIdxDict))
        print()
        count+=1

    print "------------------------------------------------------"
    print "Trigrams"
    print "------------------------------------------------------"
    count = 0
    while(count<5):
        print(makeTrigramSentence(trigrams, bigramIdxDict, wordIdxDict))
        print()
        count+=1

    print "------------------------------------------------------"
    print "Quadgrams"
    print "------------------------------------------------------"
    count = 0
    while(count<5):
        print(makeQuadgramSentence(quadgrams, trigramIdxDict, wordIdxDict))
        print()
        count+=1


if __name__ == '__main__':
    main()
