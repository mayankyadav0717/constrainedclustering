def get_document_vector(nlp, text):
    doc = nlp(text)
    word_embeddings_sum = sum(token.vector for token in doc if token.has_vector)
    document_vector = word_embeddings_sum / len(doc)
    return document_vector
