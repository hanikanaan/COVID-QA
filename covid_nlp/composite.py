from eval import eval_question_similarity
import pandas as pd
import eval
import modeling.tfidf, modeling.transformer

class Question:
    
    def __init__(self):
        pass
    
    filepath = '../COVID-QA/data/faqs/'
    with open(filepath,'faq_covidbert.csv') as f:
        qfile = pd.read_csv('faq_covidbert.csv', sep=',')
    curr = qfile.columns[0]
        
    # Recursive call: traverses the tree of questions (based on similarity) to see if any of these questions have similarity scores close to one another
    def FindQuestion(question):
        if (curr.eval_question_similarity() - question.eval_question_similarity() <= 0.85):
            return qfile[1]
        elif (curr.eval_question_similarity() - question.eval_question_similarity() > 0.85):
            qfile.columns[0].remove(curr)
            FindQuestion(question)
        return None
    
    # Traverse down the tree and order all questions in similarity if there are no similar questions that were found in the above method.
    # The tree will work such that the leaves can be considered as IDs for questions, and the more similar questions will be grouped together
    # Recursive structure helps the program traverse down the tree with logarithmic time due to the tree properties which will end up
    # helping with performance in both time and space complexities.
    def placeQuestion(question):
        min = 1
        tree = {
            {

            }
        }
        if (FindQuestion(question) == None):
            qfile.sort_values
            for i in range(0, qfile.columns[0]):
                for j in range(qfile.columns[0], i):
                    if (qfile.columns[i].eval_question_similarity() - qfile.columns[j].eval_question_similarity() <= 0.85):
                        tree.add(qfile.columns[i], qfile.row[i])
                        tree.add(qfile.columns[j], qfile.columns[j])
            for i in range (0, qfile.columns[0])
                curr = question.eval_question_similarity
                if (curr - qfile.columns[i].eval_question_similarity() < min):
                    curr_min_question_leaf = qfile.column[i]
                    min = curr - qfile.columns[i].eval_question_similarity()
    return None
