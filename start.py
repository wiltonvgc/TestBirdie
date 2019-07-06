#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np  
import re  
import nltk  
from sklearn.datasets import load_files  
nltk.download('stopwords') 
from nltk.corpus import stopwords  
import pickle  
import csv
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd


#Load DataSet => x  : data (titles) and y : target (0 or 1)
def LoadDataSet(file_path):
        smart_data = load_files(file_path)  
        x, y = smart_data.data, smart_data.target      
        
        return (x,y)  


def CreateBagWords(words):
        vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('portuguese'))  
        x = vectorizer.fit_transform(words).toarray()  

        tfidfconverter = TfidfTransformer()  
        x_tfid = tfidfconverter.fit_transform(x).toarray()  
        
        return (x_tfid,vectorizer)

def SplitDataSet(x,y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 
        
        return (x_train, x_test, y_train, y_test)

def TrainClassifier(x_train,y_train):
        classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
        classifier.fit(x_train, y_train)  
        return classifier


def ClassTitle(classifier,count_vect,title):
        return classifier.predict(count_vect.transform([title]))


def PrintReport(classifier,x_test,y_test):
        y_pred = classifier.predict(x_test)
        
        print(confusion_matrix(y_test,y_pred))  
        print(classification_report(y_test,y_pred))  
        print(accuracy_score(y_test, y_pred)) 


def ClassFileInput(classifier,count_vect,file_path):
        file_read = pd.read_csv(file_path, sep='\t')

        with open('outputClassification.csv','w') as outputFile:        
                for title in file_read['TITLE']:
                        classif = ClassTitle(classifier,count_vect,title)
                        classe = 'sim'
                        if(classif[0] == 0):
                                classe = 'não'
                        outputFile.write(title + ';' + classe)
                        outputFile.write('\n')

        outputFile.close()

def main():
        load = LoadDataSet('TrainSmartphone')
        x_tfid,vect = CreateBagWords(load[0])
        y = load[1]
        split = SplitDataSet(x_tfid,y)


        x_train = split[0]
        y_train = split[2]
        classifier = TrainClassifier(x_train,y_train)

        
        x_test = split[1]
        y_test = split[3]
        #PrintReport(classifier,x_test,y_test)

        ClassFileInput(classifier,vect,'data_estag_ds.tsv')


        


       
    
main()





























'''
arquivo = open('TrainSmartphone/no/train_no.csv')

linhas = csv.reader(arquivo)

i = 0



for linha in arquivo:
        with open('TrainSmartphone/no/title' + str(i) + '.csv','w') as textFile:
              textFile.write(linha)
        i+=1
'''






