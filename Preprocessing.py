import os
import spacy
from spacy.matcher import PhraseMatcher
import pickle
import numpy as np
import pandas as pd
from networkx import Graph, DiGraph, descendants, shortest_path
import tensorflow as tf
import tensorflow_hub as hub


# part of speech dict
# https://universaldependencies.org/u/pos/
POS = {'adj': 0, 'adp': 1, 'adv': 2, 'aux': 3, 'cconj': 4, 'conj': 5, 'det': 6, 'intj': 7, 'noun': 8,
       'num': 9, 'part': 10, 'pron': 11, 'propn': 12, 'punct': 13, 'sconj': 14, 'sym': 15, 'verb': 16, 'x': 17}
# dependency dict
# https://universaldependencies.org/u/dep/
DEP = {'acl': 0, 'acomp': 1, 'advcl': 2, 'advmod': 3, 'agent': 4, 'amod': 5, 'appos': 6, 'attr': 7, 'aux': 8, 'auxpass': 9, 'case': 10, 'cc': 11, 'ccomp': 12, 'complm': 13, 'compound': 14, 'conj': 15, 'csubj': 16, 'csubjpass': 17, 'dative': 18, 'dep': 19, 'det': 20, 'dobj': 21, 'expl': 22, 'hmod': 23, 'hyph': 24, 'infmod': 25, 'intj': 26, 'iobj': 27, 'mark': 28, 'meta': 29,
       'neg': 30, 'nmod': 31, 'nn': 32, 'nounmod': 33, 'npadvmod': 34, 'npmod': 35, 'nsubj': 36, 'nsubjpass': 37, 'num': 38, 'number': 39, 'nummod': 40, 'oprd': 41, 'parataxis': 42, 'partmod': 43, 'pcomp': 44, 'pobj': 45, 'poss': 46, 'possessive': 47, 'preconj': 48, 'predet': 49, 'prep': 50, 'prt': 51, 'punct': 52, 'quantmod': 53, 'rcmod': 54, 'relcl': 55, 'root': 56, 'xcomp': 57}
EMBEDDING_SIZE = 1100
NLP = spacy.load("en_core_web_sm")
ALL_WORD_EMBEDDING_MAP = {}

TF_GRAPH = tf.Graph()
with TF_GRAPH.as_default():
    text_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
    WORD_MODEL = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    embedded_text = WORD_MODEL(text_input)
    init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
TF_GRAPH.finalize()

def path_embedding(edges,hyponym,hypernym, session):
    path_emb = np.zeros((edges.shape[0], EMBEDDING_SIZE))
    for i, edge in enumerate(edges):
        if edge[0] not in ALL_WORD_EMBEDDING_MAP.keys() or edge[3] not in ALL_WORD_EMBEDDING_MAP.keys():
            result = session.run(embedded_text, feed_dict={text_input: [edge[0], edge[3]]})
            ALL_WORD_EMBEDDING_MAP[edge[0]] = result[0]
            ALL_WORD_EMBEDDING_MAP[edge[3]] = result[1]
        x = ALL_WORD_EMBEDDING_MAP[edge[0]]
        y = ALL_WORD_EMBEDDING_MAP[edge[3]]

        if(edge[0] == hyponym or edge[0] == hypernym):
            x = np.ones(x.shape) * 0.0000001
        if(edge[3] == hyponym or edge[3] == hypernym):
            y = np.ones(y.shape) * 0.0000001

        pos = np.zeros((len(POS),))
        pos[POS[edge[1]]] = 1.0
        dep = np.zeros((len(DEP),))
        dep[DEP[edge[2]]] = 1.0
        path_emb[i] = np.hstack([x, pos, dep, y])

    return path_emb


def parse_sentence_old(x, y, sentence):
    doc = NLP(u"" + sentence)

    # Find x and y phrase chunks in sentence
    matcher = PhraseMatcher(NLP.vocab, attr="LOWER")
    matcher.add("X", None, NLP(x), NLP(x + "s"))
    matcher.add("Y", None, NLP(y), NLP(y + "s"))
    matches = matcher(doc)

    # Delete overlapping matches
    del_idx = []
    for i in range(len(matches)):
        (match_id_i, start_i, end_i) = matches[i]
        for j in range(i + 1, len(matches)):
            (match_id_j, start_j, end_j) = matches[j]
            if end_i >= start_j and end_i >= end_j:
                del_idx.append(j)
            elif end_i >= start_j and end_i <= end_j:
                if end_i - start_i >= end_j - start_j:
                    del_idx.append(j)
                else:
                    del_idx.append(i)
            else:
                pass
    matches = [match for idx, match in enumerate(matches) if idx not in del_idx]
    matches = sorted(matches, key=lambda z: z[1], reverse=True)

    # Choose one chunk for x and one chunk for y
    seen = set()
    matches = [(a, b, c) for a, b, c in matches if not (a in seen or seen.add(a))]
    assert len(matches) == 2

    # Merge x and y chunks
    for (match_id, start, end) in matches:
        string_id = NLP.vocab.strings[match_id]
        if string_id == "X":
            x_span = doc[start:end]
            x_span.merge(x_span.root.tag_, x_span.root.lemma_,
                         x_span.root.ent_type_)
        else:
            y_span = doc[start:end]
            y_span.merge(y_span.root.tag_, y_span.root.lemma_,
                         y_span.root.ent_type_)

    # Get all edges
    edges = []
    for token in doc:
        for child in token.children:
            edges.append((token.lower_,
                          token.pos_.lower(),
                          child.dep_,
                          child.lower_))

    return np.array(edges)


def parse_sentence_new(x, y, sentence):
    doc = NLP(u"" + sentence)

    # Find x and y phrase chunks in sentence
    matcher = PhraseMatcher(NLP.vocab, attr="LOWER")
    matcher.add("X", None, NLP(x), NLP(x + "s"))
    matcher.add("Y", None, NLP(y), NLP(y + "s"))
    matches = matcher(doc)

    # Delete overlapping matches
    del_idx = []
    for i in range(len(matches)):
        (match_id_i, start_i, end_i) = matches[i]
        for j in range(i + 1, len(matches)):
            (match_id_j, start_j, end_j) = matches[j]
            if end_i >= start_j and end_i >= end_j:
                del_idx.append(j)
            elif end_i >= start_j and end_i <= end_j:
                if end_i - start_i >= end_j - start_j:
                    del_idx.append(j)
                else:
                    del_idx.append(i)
            else:
                pass
    matches = [match for idx, match in enumerate(matches) if idx not in del_idx]
    matches = sorted(matches, key=lambda z: z[1], reverse=True)

    # Choose one chunk for x and one chunk for y
    seen = set()
    matches = [(a, b, c) for a, b, c in matches if not (a in seen or seen.add(a))]
    assert len(matches) == 2

    # Merge x and y chunks
    for (match_id, start, end) in matches:
        string_id = NLP.vocab.strings[match_id]
        if string_id == "X":
            x_span = doc[start:end]
            x_span.merge(x_span.root.tag_, x_span.root.lemma_,
                         x_span.root.ent_type_)
        else:
            y_span = doc[start:end]
            y_span.merge(y_span.root.tag_, y_span.root.lemma_,
                         y_span.root.ent_type_)

    # Track x and y chunks
    x = x_span.lower_ + str(x_span.start)
    y = y_span.lower_ + str(y_span.start)

    #  Get directed and undirected graphs
    graph_edges = []
    for token in doc:
        for child in token.children:
            graph_edges.append((token.lower_ + str(token.i),
                                child.lower_ + str(child.i)))
    directed_graph = DiGraph(graph_edges)
    undirected_graph = Graph(graph_edges)

    # Shortest path between x and y
    p = []
    sp = shortest_path(undirected_graph, source=x, target=y)
    for token in doc:
        for child in token.children:
            if token.lower_ + str(token.i) in sp and child.lower_ + str(child.i) in sp:
                p.append((token.lower_ + str(token.i),
                          child.lower_ + str(child.i)))

    # Descendants of x and y
    xd = sorted(descendants(directed_graph, x), key=lambda z: int(z[-1]))
    yd = sorted(descendants(directed_graph, y), key=lambda z: int(z[-1]))
    for v, d in [(x, xd), (y, yd)]:
        for desc in d:
            vp = shortest_path(directed_graph, source=v, target=desc)
            for wi, wj in zip(vp, vp[1:]):
                if (wi, wj) not in p:
                    p.append((wi, wj))

    # Gather edges with POS and DEP labels
    edges = []
    for edge in p:
        for token in doc:
            for child in token.children:
                if edge == (token.lower_ + str(token.i), child.lower_ + str(child.i)):
                    edges.append((token.lower_,
                                  token.pos_.lower(),
                                  child.dep_,
                                  child.lower_))

    return np.array(edges)


def get_embeddings(dataset):

    file_path = os.path.abspath("")
    # triplets = generate_triplets()
    triplets = pd.read_csv(os.path.join(file_path, "dataset", dataset +".csv")).values

    #with open(os.path.join(file_path, "dataset", "our_dataset_dict.pkl"), "rb") as fd:
    #    ALL_WORD_EMBEDDING_MAP = pickle.load(fd)

    all_words = []

    for i, triplet in enumerate(triplets):
        print(str(i + 1) + "/" + str(triplets.shape[0]), end="\r")
        try:
            if "\r" in triplet[2]:
                triplet[2] = triplet[2].replace("\r", "")
            if "\n" in triplet[2]:
                triplet[2] = triplet[2].replace("\n", "")
            edges = parse_sentence_old(triplet[0], triplet[1], triplet[2])
            for edge in edges:
                all_words.append(edge[0])
                all_words.append(edge[3])
        except Exception as e:
            pass
    all_words = np.unique(all_words)
    session = tf.compat.v1.Session(graph=TF_GRAPH)
    session.run(init_op)
    result = session.run(embedded_text, feed_dict={
                         text_input: all_words})
    for i, word in enumerate(all_words):
        ALL_WORD_EMBEDDING_MAP[word] = result[i]

    data_old_x = []
    data_new_x = []
    data_y = []
    session = tf.compat.v1.Session(graph=TF_GRAPH)
    session.run(init_op)
    new_paths = []

    xs = []
    ys= []
    sentences = []

    for i, triplet in enumerate(triplets):
        print(str(i + 1) + "/" + str(triplets.shape[0]), end="\r")
        try:
            if "\r" in triplet[2]:
                triplet[2] = triplet[2].replace("\r", "")
            if "\n" in triplet[2]:
                triplet[2] = triplet[2].replace("\n", "")
            #edges_old = parse_sentence_old(triplet[0], triplet[1], triplet[2])
            #path_old = path_embedding(edges_old,session)
            edges_new = parse_sentence_new(triplet[0], triplet[1], triplet[2])
            path_new = path_embedding(edges_new,triplet[0],triplet[1],session)

            new_paths.append(edges_new)
            data_new_x.append(path_new)
            if(triplet[4] == True):
                data_y.append(1.0)
            else:
                data_y.append(0.0)

            xs.append(triplet[0])
            ys.append(triplet[1])
            sentences.append(triplet[2])


        except Exception as e:
            print(triplet)
            print(e)
            pass

    with open(os.path.join(file_path, "dataset", dataset+"_x.pkl"), "wb") as fx:
        pickle.dump(data_new_x, fx)
    with open(os.path.join(file_path, "dataset", dataset+"_paths_x.pkl"), "wb") as fx:
        pickle.dump(new_paths, fx)
    with open(os.path.join(file_path, "dataset", dataset+"_y.pkl"), "wb") as fy:
        pickle.dump(data_y, fy)
    with open(os.path.join(file_path, "dataset", "our_dataset_dict.pkl"), "wb") as fd:
        pickle.dump(ALL_WORD_EMBEDDING_MAP, fd)

    df = pd.DataFrame(data = {'x': xs, 'y': ys, 'sentences' : sentences})
    df.to_csv(os.path.join(file_path, "dataset", dataset+"_dataset.csv"), index = False, columns = None)
