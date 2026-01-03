import os
import json
from dotenv import load_dotenv
load_dotenv()
os.environ["JAVA_HOME"] = os.getenv("JAVA_HOME")
from tqdm import tqdm
from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
import pickle

class SearchEngine:
    def __init__(self, approach="qld"):
        self.reader = IndexReader.from_prebuilt_index('robust04')
        self.searcher = LuceneSearcher.from_prebuilt_index('robust04')
        if approach=="qld":
            self.searcher.set_qld(mu=1000)
        # Other approaches here

    def get_top_k(self, query, k=5, to_chunk=False):
        """
        Get the top k ranked (full) documents using the searcher
        :param query: the query
        :param k: top results to retrieve (default: 5)
        :param to_chunk: To chunk or not to chunk (default: False for now)
        :return:
        """
        context = []
        context_metadatas = [] # TODO look for metadata
        hits = self.searcher.search(query, k)
        # Get text from hits
        for hit in hits:
            doc = self.searcher.doc(hit.docid)
            raw_doc = doc.raw()
            # TODO metadata
            # metadata = {'wiki_id':data.get('wikipedia_id', ''),
            #             'title': data.get('wikipedia_title', ''),
            #             'categories': data.get('categories', ',').split(','),}
            # context_metadatas.append(metadata)
            # TODO proper parsing
            context.append(raw_doc)
        if not to_chunk:
            return context, context_metadatas
        else: # Passage retrieval stuff
            # all_paragraphs = clean_split_wiki_docs(context, context_metadatas)
            #return all_paragraphs
            return context, context_metadatas

    def get_context_batch(self, samples, k=5, save_name=None):
        all_contexts = {}
        for query in tqdm(samples['question']):
            all_contexts[query] = self.get_top_k(query, k, to_chunk=False)
        if save_name:
            with open(f'top{k}_full_docs{save_name}.pkl', 'wb') as f:
                pickle.dump(all_contexts, f)
        else:
            print("Did not provide save_name, no pkl will be created")
        return all_contexts
