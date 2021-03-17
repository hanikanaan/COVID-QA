import covid_nlp.*

class Question():

    __init__()

    def search(search_string): 
        input = input()
        lang = LanguageDetector.detect_lang_sil(input, input)                  
        #language detector is called so that the input language is detected
        search_string = MSTranslator(self, None, None, lang).translate(self, input) 
        path = 'questions-aggregate\data\question-answering'
        with open(path, 'COVID-QA.json') as f:
            resu = f.search(search_string)
        if (LanguageDetector.detect_lang_sil(resu, resu) != lang)
            resu = MSTranslator(self, None, None, LanguageDetector.detect_lang_sil(resu, resu).translate(self, resu))
        return resu


# This aggregate is made to compile all the different functions that are needed to ask for a question for input, get the input, detect the language in which the input is, translate
# to the data set language, matching the question to one of the keys in the dictionary in the COVID-QA json file, and then returning and translating (if needed).