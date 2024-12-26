import nltk
nltk.download('punkt')

def remove_zero_val_words(tfidf_df, document_index, threshold=0.0001):
    if document_index < 0 or document_index >= len(tfidf_df):
        raise ValueError("Invalid document_index: it should be within the range of the DataFrame.")
    row = tfidf_df.iloc[document_index]
    filtered_row = row[row > threshold]
    words = filtered_row.index.tolist()
    return words


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def generate_constraints(keyphrase_list, dataframe, similarity_matrix):
    must_link = []
    cannot_link = []

    for i in range(len(dataframe)):
        for j in range(i + 1, len(dataframe)):
            similarity_score = jaccard_similarity(set(keyphrase_list[i]), set(keyphrase_list[j]))
            if similarity_score > 0.1 and  similarity_matrix[i, j] > 0.3:
                must_link.append((i, j))
    for i in range(len(keyphrase_list)):
        for j in range(i + 1, len(keyphrase_list)):
            similarity_score = jaccard_similarity(set(keyphrase_list[i]), set(keyphrase_list[j]))
            if similarity_score <= 0 and  similarity_matrix[i, j] < 0.3:
                cannot_link.append((i, j))
    return must_link, cannot_link

def iterative_dfs(v, graph, visited, component):
    stack = [v]
    while stack:
        node = stack.pop()
        if not visited[node]:
            visited[node] = True
            component.append(node)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)

def transitive_entailment_graph(ml, cl, dslen):
    ml_graph = {}
    cl_graph = {}
    for i in range(dslen):
        ml_graph[i] = set()
        cl_graph[i] = set()

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    visited = [False] * dslen
    neighborhoods = []
    for i in range(dslen):
        if not visited[i]:
            component = []
            iterative_dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
            neighborhoods.append(component)

    for (i, j) in cl:
            cl_graph[i].add(j)
            cl_graph[j].add(i)
            for y in ml_graph[j]:
                if y not in ml_graph[i]:
                    cl_graph[i].add(y)
                    cl_graph[y].add(i)
            for x in ml_graph[i]:
                if x not in ml_graph[j]:
                    cl_graph[x].add(j)
                    cl_graph[j].add(x)
                    for y in ml_graph[j]:
                        if y not in ml_graph[x]:
                            cl_graph[x].add(y)
                            cl_graph[y].add(x)
    return neighborhoods, ml_graph, cl_graph
