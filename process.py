import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('wordnet')
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


class PreProcess: 
    def __init__(self, documents):
        self.documents = documents
        self.stopwords = set(stopwords.words('english'))
        self.porter_stemmer = PorterStemmer()
        self.wordnet_lemmatizer = WordNetLemmatizer()

    def dataframe(self):
        if isinstance(self.documents, dict):
            df = pd.DataFrame(list(self.documents.items()), columns=["docno", "doc"])
        else:
            df = self.documents
        return df
    
    def process(self):
        df = self.dataframe()
        df['doc'] = df['doc'].apply(self.preprocessData)
        return df
    
    def preprocessData(self,text):
        header_pattern = re.compile(r'^(From|Subject|Nntp-Posting-Host|Organization|Lines|Summary|Keywords|Distribution|Article-I.D.):.*$', re.MULTILINE)
        text = re.sub(r'\d+', '', text)
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        text = text+' '.join(filtered_words)
        ps = PorterStemmer()
        text_no_headers = re.sub(header_pattern, '', text)
        parts = text_no_headers.split('\n\n', 1)
        if len(parts) > 1:
            main_content = parts[1]
        else:
            main_content = parts[0]  
        main_content = main_content.translate(str.maketrans('', '', string.punctuation))
        main_content = main_content.strip()
        main_content = ps.stem(main_content)
        return main_content;

def remove_stopWords(text):
    text = re.sub(r'\d+', '', text)
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


