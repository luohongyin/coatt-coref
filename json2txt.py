import sys
import json

def plain_txt(s):
    p = []
    for ss in s['sentences']:
        p += ss
    return ' '.join(p)

if __name__ == '__main__':
    fn = '%s.english.jsonlines' % sys.argv[1]
    lines = open(fn, 'r').readlines()
    out_handle = open('%s.english.txt' % sys.argv[1], 'w')
    for line in lines:
        out_handle.write(plain_txt(json.loads(line)).encode('utf-8') + '\n')
    out_handle.close()
