import sys
import json

def list_min(l):
    return min([item[0] for item in l])

def sort_cluster(line):
    example = json.loads(line)
    cluster = example['clusters']
    sorted_cluster = sorted(cluster, key = lambda x: list_min(x), reverse=False)
    example['clusters'] = sorted_cluster
    return json.dumps(example)

if __name__ == '__main__':
    input_file = '%s.english.jsonlines' % sys.argv[1]
    output_file = '%s_sorted.english.jsonlines' % sys.argv[1]
    data = open(input_file, 'r').readlines()
    new_data = [sort_cluster(x) for x in data]
    open(output_file, 'w').write('\n'.join(new_data))
