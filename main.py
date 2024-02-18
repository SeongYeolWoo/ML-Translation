from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import translation
from utils.preprocessing import detokenize, en_tokenize, en_bpe

app = FastAPI()

class Sentence(BaseModel):
    sentence: str = Field(title='영어 문장')

    model_config = {
        'json_schema_extra':{
            'examples':[
                {'sentence' : "Her strange idea approaches to me interestingly."}
            ]
        }
    }
class ErrorMessage(BaseModel):
    code: int
    sentence: str
    message: str

class TranslateMessage(BaseModel):
    code: int
    English: str
    Korean: str



# Seq2Seq 모델
@app.get('/seq2seq', description="Seq2Seq 영한 번역기 API 입니다.", response_model=Union[TranslateMessage,ErrorMessage])
async def seq2seq(sentence):

    try:
        result = en_tokenize(sentence)
    except Exception as e:
        return {
            'code': 401,
            'sentence': sentence,
            'message': '토크나이징 에러'
        }
    
    try:
        result = en_bpe(result)

    except Exception as e:
        print(e)
        return {
            'code': 402,
            'sentence': sentence,
            'message': 'BPE 에러'
        }
    print(result)
    try:
        result = translation(result)
        print(result)
    except Exception as e:
        print(e)
        return {
            'code': 403,
            'sentence': sentence,
            'message': '모델 에러'
        }
    try:
        result = detokenize(result)

    except Exception as e:
        print(e)
        return {
            'code': 404,
            'sentence': sentence,
            'message': '토큰화 해제 중 에러'
        }
    
    return {
        'code': 200,
        'English': sentence,
        'Korean': result
    }
        


if __name__=='__main__':
    print(translation('Music is often a mix of pop, hip-hop, and many other genres, and is more comfortable than electronic sounds, which are often heard on German charts.'))