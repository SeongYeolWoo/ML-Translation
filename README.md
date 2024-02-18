# ML-Translation

## 훈련시 train.py 
- tokenize 후 BPE 적용한 데이터셋으로 학습
- BPE는 김기현님의 subword-nmt를 사용하였습니다.(https://github.com/kh-kim/subword-nmt)
- 데이터셋: AIHUB 한국어-영어 변역(병렬) 말뭉치(https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=126)
```
train.py [-h] --model_fn MODEL_FN --train TRAIN --valid VALID --lang LANG [--gpu_id GPU_ID]
                [--off_autocast] [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS] [--verbose VERBOSE]
                [--init_epoch INIT_EPOCH] [--max_length MAX_LENGTH] [--dropout DROPOUT]
                [--word_vec_size WORD_VEC_SIZE] [--hidden_size HIDDEN_SIZE] [--n_layers N_LAYERS]
                [--max_grad_norm MAX_GRAD_NORM] [--iteration_per_update ITERATION_PER_UPDATE] [--lr LR]
                [--lr_step LR_STEP] [--lr_gamma LR_GAMMA] [--lr_decay_start LR_DECAY_START] [--use_adam]
                [--use_radam] [--rl_lr RL_LR] [--rl_n_samples RL_N_SAMPLES] [--rl_n_epochs RL_N_EPOCHS]
                [--rl_n_gram RL_N_GRAM] [--rl_reward RL_REWARD] [--use_transformer]
                [--n_splits N_SPLITS]
```

## FastAPI 배포
```
uvicorn main:app --reload --host <host ip> --port <post>
```
```
<host ip>:<port>/seq2seq?sentence={sentence}
```
