from eval import eval_question_similarity
import pandas as pd
import eval
import modeling.tfidf, modeling.transformer

class Question:
    
    def __init__(self):
        pass

        
    # Recursive call: traverses the tree of questions (based on similarity) to see if any of these questions have similarity scores close to one another
    def FindQuestion(question):  
        filepath = '../COVID-QA/data/faqs/'
        with open(filepath,'faq_covidbert.csv') as f:
            qfile = pd.read_csv('faq_covidbert.csv', sep=',')
        curr = qfile.columns[0]
        if (curr.eval_question_similarity() - question.eval_question_similarity() <= 0.85):
            return qfile[1]
        elif (curr.eval_question_similarity() - question.eval_question_similarity() > 0.85):
            qfile.columns[0].remove(curr)
            FindQuestion(question)
        return None
    

    def placeQuestion(question):
        tree = {
            {

            }
        }
        if (FindQuestion(question) == None):
            filepath = '../COVID-QA/data/faqs/'
            with open(filepath, 'faq_covidbert.csv') as f:
                qfile = pd.read_csv('faq_covidbert.csv', sep=',')
            qfile.sort_values
            for i in range(0, qfile.columns[0]):
                for j in range(qfile.columns[0], i):
                    if (qfile.columns[i].eval_question_similarity - qfile.columns[j].eval_question_similarity <= 0.85):
                        tree.add(qfile.columns[i], qfile.row[i])
                        tree.add(qfile.columns[j], qfile.columns[j])
