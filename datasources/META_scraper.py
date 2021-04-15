import importlib.util
import logging
import os

import pandas as pd
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.elasticsearch import ElasticsearchRetriever
from scrapy.crawler import CrawlerProcess

logger = logging.getLogger(__name__)

PATH = os.getcwd() + "/scrapers"
RESULTS = []
MISSED = []

class ScraperPrototypeInterface:
  @abstractstaticmethod
  def clone():

class CovidScraper(METACLASS=ScraperPrototypeInterface):
    
    def __init__(self, name, start_urls):
        self.name = name
        self.start_urls = start_urls
    
    def clone():
        return type(self) (
            self.name.copy()
            self.start_urls.copy()
        )
    
    def parse(self, response):
        columns = {
            "question" : [],
            "answer" : [],
            "answer_html" : [],
            "link" : [],
            "name" : [],
            "category" : [],
            "last_update" : [],
        }

        current_category = ""
        current_question = ""
        current_answer = ""
        current_answer_html = ""
        ba_content_article_count = 0

        all_nodes = response.xpath("//*")
        for node in all_nodes:
            if node.attrib.get("class") == "ba-content-row":
                ba_content_article_count += 1
                # end of FAQ 
                if ba_content_article_count == 4:
                    break

            # in question
            if node.attrib.get("class") == "collapsed":
                # save previous question-answer pair
                if current_question:
                    columns["question"].append(current_question)
                    columns["answer"].append(current_answer)
                    columns["answer_html"].append(current_answer_html)
                current_question = node.css("::text").get().strip()
                continue

            # in answer
            if node.attrib.get("class") == "ba-copytext":
                current_answer = node.css(" ::text").getall()
                current_answer = " ".join(current_answer).strip()
                current_answer_html = node.getall()
                current_answer_html = " ".join(current_answer_html).strip()
                continue



        columns["question"].append(current_question)
        columns["answer"].append(current_answer)
        columns["answer_html"].append(current_answer_html)

        today = date.today()

        columns["link"] = [self.start_urls] * len(columns["question"])
        columns["name"] = ["FAQ: Corona-Virus"] * len(columns["question"])
        columns["category"] = [""] * len(columns["question"])
        columns["last_update"] = [today.strftime("%Y/%m/%d")] * len(columns["question"])

        return columns

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
