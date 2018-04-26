import sys
import json

def flatten(l):
    return [item for sublist in l for item in sublist]

def get_mentions(line):
    sample = json.loads(line)
    return flatten(sample['clusters'])

def check_line(line, target):
    mentions = get_mentions(line)
    starts = [x[target] for x in mentions]
    overlap = len(starts) - len(set(starts))
    if overlap:
        print sorted(mentions, key = lambda x: x[target], reverse=False), overlap
    return overlap > 0

def check_lines(lines, target):
    return [check_line(x, target) for x in lines]

if __name__ == '__main__':
    data = open('%s.english.jsonlines' % sys.argv[1], 'r').readlines()
    num_sample = len(data)
    print sum(check_lines(data, int(sys.argv[2]))), num_sample