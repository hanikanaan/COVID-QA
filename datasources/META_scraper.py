import importlib.util
import logging
import os

from covid_nlp.language import detect_language as match
import pandas as pd
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.elasticsearch import ElasticsearchRetriever
from scrapy.crawler import CrawlerProcess

logger = logging.getLogger(__name__)

PATH = os.getcwd() + "/scrapers"
RESULTS = []
MISSED = []

# Scrapers can be bundled together as an aggregate in their own 
# Doing so is beneficial since there are several different scrapers depending on the data source, so rather than having
# a call to each scraper, it could be made such that the system "decides" what scraper is most suitable depending on the
# question that the individual has asked (depending on any keywords such as location, language, statistic, etc.)
# Since this would require some NLP and ML models to assist in accurate matching, those parts will be done in
# pseudocode.


question = input()


# Assume there is a getDataSources algorithm that finds an appropriate data source depending on the information provided
class Matching():
    
    datasources = []

    def __init__(self):
        pass

    def match_language(self):
        language = match.LanguageDetector.detect_lang_sil(__self__, question)
        # pseudocode
        datasources.append(language.getDataSources())
        return datasources
    
    def match_location(self):
        location = input()
        # pseudocode:
        datasources.append(location.getDataSources())
        return datasources
    
    def match_statistic(self):
        age = input(int)
        demographic = input()
        profession = input()
        region_of_residence = input()


        information = {
            age,
            profession,
            region_of_residence
        }
        #pseudocode
        if (age in A): # A is an arbitrary age range
            datasources.append(age.getDataSources())
        if (profession is B): # B is an arbitrary career/profession
            datasources.append(profession.getDataSources())
        if (region_of_residence in C): # C is an arbitrary residential area
            datasources.append(profession.getDataSources())
        
    def optimal_sources(self):
        return datasources


class Pipeline(object):
    questionsOnly = True

    def filter(self, item, index):
        question = item['question'][index].strip()
        if self.questionsOnly and not question.endswith("?"):
            return False
        if len(item['answer'][index].strip()) == 0:
            return False
        return True

    def process_item(self, item, spider):
        if len(item['question']) == 0:
            logger.error("Scraper '" + spider.name + "' provided zero results!")
            MISSED.append(spider.name)
            return
        validatedItems = {}
        for key, values in item.items():
            validatedItems[key] = []
        for i in range(len(item['question'])):
            if not self.filter(item, i):
                continue
            for key, values in item.items():
                validatedItems[key].append(values[i])
        if len(validatedItems['question']) == 0:
            logger.error("Scraper '" + spider.name + "' provided zero results after filtering!")
            MISSED.append(spider.name)
            return
        df = pd.DataFrame.from_dict(validatedItems)
        RESULTS.append(df)


if __name__ == "__main__":
    logging.disable(logging.WARNING)

    crawler_files = [os.path.join(PATH, f) for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f)) and (not f.startswith('.'))]
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
        'ITEM_PIPELINES': {'__main__.Pipeline': 1}
    })
    for crawler in crawler_files:
        scraper_spec = importlib.util.spec_from_file_location("CovidScraper", crawler)
        scraper = importlib.util.module_from_spec(scraper_spec)
        scraper_spec.loader.exec_module(scraper)
        CovidScraper = scraper.CovidScraper
        process.crawl(CovidScraper)
    process.start()
    dataframe = pd.concat(RESULTS)
    dataframe.fillna(value="", inplace=True)
    dataframe["answer"] = dataframe['answer'].str.strip()
    if len(MISSED) > 0:
        logger.error(f"Could not scrape: {', '.join(MISSED)} ")

    MODEL = "bert-base-uncased"
    GPU = False
    document_store = ElasticsearchDocumentStore(
        host="localhost",
        username="",
        password="",
        index="document",
        text_field="answer",
        embedding_field="question_emb",
        embedding_dim=768,
        excluded_meta_data=["question_emb"],
    )

    retriever = ElasticsearchRetriever(document_store=document_store, embedding_model=MODEL, gpu=GPU)

    dataframe.fillna(value="", inplace=True)
    # Index to ES
    docs_to_index = []

    for doc_id, (_, row) in enumerate(dataframe.iterrows()):
        d = row.to_dict()
        d = {k: v.strip() for k, v in d.items()}
        d["document_id"] = doc_id
        # add embedding
        question_embedding = retriever.create_embedding(row["question"])
        d["question_emb"] = question_embedding
        docs_to_index.append(d)
    document_store.write_documents(docs_to_index)
