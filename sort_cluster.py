import sys
import util
import json
import numpy as np

def list_min(l):
    return min(l)

def sort_cluster(line):
    example = json.loads(line)
    cluster = example['clusters']
    sorted_cluster = sorted(cluster, key = lambda x: list_min(x), reverse=False)
    example['clusters'] = sorted_cluster
    return json.dumps(example)

def sort_all_samples():
    input_file = '%s.english.jsonlines' % sys.argv[1]
    output_file = '%s_sorted.english.jsonlines' % sys.argv[1]
    data = open(input_file, 'r').readlines()
    new_data = [sort_cluster(x) for x in data]
    open(output_file, 'w').write('\n'.join(new_data))

def get_mention_dict(clusters):
    mention_dict = {}
    for i, cluster in enumerate(clusters):
        for mention in cluster:
            mention_dict[tuple(mention)] = i + 1
    return mention_dict

def indices(length, sen_index):
    spans = []
    for i in range(length):
        for j in range(10):
            if i - j < 0:
                break
            if sen_index[i] != sen_index[i - j]:
                break
            spans.append((i - j, i))
    return spans

def get_value(d, x):
    if x in d:
        return d[x]
    else:
        return 0

def get_sen_index(sentences):
    start = 0
    sen_index = {}
    for i, sen in enumerate(sentences):
        for j, word in enumerate(sen):
            sen_index[start + j] = i
        start += len(sen)
    return sen_index

def process_line(line):
    example = json.loads(line)
    clusters = example['clusters']
    sentences = util.flatten(example['sentences'])
    sen_index = get_sen_index(example['sentences'])
    text_length = len(sentences)
    mention_dict = get_mention_dict(clusters)
    spans = indices(text_length, sen_index)
    example['span_labels'] = [get_value(mention_dict, x) for x in spans]
    return json.dumps(example)

def span_labels():
    input_file = '%s_sorted.english.jsonlines' % sys.argv[1]
    output_file = '%s_sl.english.jsonlines' % sys.argv[1]
    data = open(input_file, 'r').readlines()
    data_labeled = [process_line(x) for x in data]
    open(output_file, 'w').write('\n'.join(data_labeled))

if __name__ == '__main__':
    # sort_all_samples()
    span_labels()
