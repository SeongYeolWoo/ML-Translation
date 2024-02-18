from konlpy.tag import Mecab
from mosestokenizer import *
from subword_nmt.apply_bpe import BPE
import subword_nmt.apply_bpe as apply_bpe
import argparse
import sys
import codecs

m = Mecab()
moses = MosesTokenizer('en')

# 원 문장 띄어쓰기 시 _ 추가
def post_tokenize(tok, sentence):
    ref = sentence.strip()
    STR = '▁'
    buf = [STR]

    tok_index = 0
    ref_index = 0
    
    while tok_index < len(tok):
        if tok != '':
            c = tok[tok_index]
            tok_index = tok_index + 1

            if c != ' ':
                while ref_index < len(ref):
                    c_ = ref[ref_index]
                    ref_index = ref_index + 1
                    
                    if c_ == ' ':
                        c = STR + c
                    else:
                        break
        buf += [c]

    return ''.join(buf)


# Korean Tokenizer
def ko_tokenize(sentence):
    tok = ' '.join(m.morphs(sentence))

    return post_tokenize(tok, sentence)
                        
# English Tokenizer
def en_tokenize(sentence):
    tok = ' '.join(moses(sentence.strip()))
    
    return post_tokenize(tok, sentence)

# BPE 수행
def en_bpe(sentence):
    bpe_model_fn = '/home/sung/NLG/bpe.en.model'
    bpe_model = open(bpe_model_fn, encoding = 'utf-8')

    parser = apply_bpe.create_parser()
    args = parser.parse_args()

    # print(args.codes, args.merges, args.separator, args.vocabulary, args.glossaries)

    # read/write files as UTF-8
    args.codes = codecs.open(bpe_model_fn, encoding='utf-8')
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    if args.vocabulary:
        args.vocabulary = codecs.open(args.vocabulary.name, encoding='utf-8')

    if args.vocabulary:
        vocabulary = apply_bpe.read_vocabulary(args.vocabulary, args.vocabulary_threshold)
    else:
        vocabulary = None

    bpe = BPE(args.codes, args.merges, args.separator, vocabulary, args.glossaries)
    
    line = sentence

    result = bpe.segment(line.strip())

    trailing_whitespace = len(line)-len(line.rstrip())
    if trailing_whitespace:
        
        result += ((line[-trailing_whitespace:]))

    return result
    # st = ''

    # for t in tmp:
    #     st += ' '.join(t)
    #     st += ' '

    # return st.strip()

def en_bpe(sentence):
    bpe_model_fn = '/home/sung/NLG/bpe.en.model'
    bpe_model = codecs.open(bpe_model_fn, encoding = 'utf-8')
    bpe = BPE(bpe_model)
    
    line = sentence
    # print(line)

    result = bpe.segment(line.strip())

    trailing_whitespace = len(line)-len(line.rstrip())
    if trailing_whitespace:
        result += ((line[-trailing_whitespace:]))

    return result                     

def preprocessing(sentence):
    return en_bpe(en_tokenize(sentence))


# ▁ 제거
def detokenize(sentence):
    if '▁▁' in sentence.strip():
        return sentence.strip().replace(' ', '').replace('▁▁', ' ').replace('▁', '').strip()
    else:
        return sentence.strip().replace(' ', '').replace('▁', ' ').strip()

if __name__ == '__main__':
    print(preprocessing('Music is often a mix of pop, hip-hop, and many other genres, and is more comfortable than electronic sounds, which are often heard on German charts.'))