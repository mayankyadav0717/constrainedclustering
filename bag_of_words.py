from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class BagOfWords:
    def __init__(self, dataframe, column_name):
        self.dataframe = dataframe
        self.column_name = column_name
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.fitted = False  

    def tf_idf(self):
        series = self.dataframe[self.column_name]
        tf_idf_matrix = self.vectorizer.fit_transform(series)
        self.fitted = True  
        tf_idf_df = pd.DataFrame(tf_idf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())
        return  tf_idf_matrix, tf_idf_df



