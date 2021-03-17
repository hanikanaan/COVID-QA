import covid_nlp.language.detect_language as detect, covid_nlp.language.ms_translate as translate
import covid_nlp.eval
import covid_nlp.modeling.tfidf, covid_nlp.modeling.transformer


class Question():

    def __init__(self):
        pass

    def search(self, search_string): 
        input = input()
        lang = detect.detect_lang_sil(input, input)                  
        #language detector is called so that the input language is detected
        search_string = translate.MSTranslator(self, None, None, lang).translate(self, input) 
        path = 'COVID-QA\data\question-answering'
        with open(path, 'COVID-QA.json') as f:
            resu = f.search(search_string)
        if (detect.LanguageDetector.detect_lang_sil(resu, resu) != lang):
            resu = translate.MSTranslator(self, None, None, detect.LanguageDetector.detect_lang_sil(resu, resu).translate(self, resu))
        return resu


# This aggregate is made to compile all the different functions that are needed to ask for a question for input, get the input, detect the language in which the input is, translate
# to the data set language, matching the question to one of the keys in the dictionary in the COVID-QA json file, and then returning and translating (if needed).