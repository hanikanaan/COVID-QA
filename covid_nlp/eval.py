import pandas as pd
import numpy as np
import covid_nlp.language.detect_language as detect, covid_nlp.language.ms_translate as translate
import covid_nlp.eval
import covid_nlp.modeling.tfidf, covid_nlp.modeling.transformer

from sklearn.metrics import roc_auc_score, f1_score
from farm.utils import MLFlowLogger


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


def eval_question_similarity(y_true, y_pred, lang, model_name, params, user=None, log_to_mlflow=True, run_name="default"):
    # basic metrics
    mean_diff = np.mean(np.abs(y_true - y_pred))
    roc_auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred.round(0))
    metrics = {"roc_auc": roc_auc, "mean_abs_diff": mean_diff, "f1_score": f1}
    print(metrics)

    # log experiment results to MLFlow (visit https://public-mlflow.deepset.ai/)
    if log_to_mlflow:
        params["lang"] = lang
        params["model_name"] = model_name
        if user:
            params["user"] = user

        ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
        ml_logger.init_experiment(experiment_name="COVID-question-sim", run_name=run_name)
        ml_logger.log_params(params)
        ml_logger.log_metrics(metrics, step=0)


if __name__ == "__main__":
    # config
    eval_file = "../data/eval_question_similarity_en.csv"
    lang = "en"
    model_name = "naive_baseline"
    experiment_name = "naive_baseline_1"
    log_to_mlflow = True
    params = {"some_model_param": 0}

    # load eval data
    df = pd.read_csv(eval_file)

    # predict similarity of samples (e.g. via embeddings + cosine similarity)
    # here: dummy preds for naive baseline
    y_true = df["similar"].values
    y_pred = [0.5] * len(y_true)

    # eval & track results
    eval_question_similarity(y_true=y_true, y_pred=y_pred, lang=lang, model_name=model_name,
                             params=params, user="malte", log_to_mlflow=log_to_mlflow, run_name=experiment_name)


