# Large Language Models are zero-shot text classifiers

The new version of code has released

### Code
#### Baseline
> python run_baseline # run ML NN baselines
> python run_bart_deberta # run bart deberta models 

#### LLMs
> python run_llms # run all LLMs, ChatGPT, Gemini, Vicuna, Qwen, etc, you can also run files simultaneously

I would suggest you to run them separately to avoid conda env conflicts
> python run_fschat # vicuda
> python run_gemini
etc

#### Finetune
##### Llama3
For details of how to finetune llama3, please checkout https://github.com/unslothai/unsloth
> python finetune_llama3 # finetune llama3-8B model
> python run_finetuned_llama3 # run for finetuned llam3-8B model, you need to adjust the model path

##### Qwen
For details of how to finetune Qwen, please checkout https://github.com/QwenLM/Qwen/blob/main/finetune/finetune_lora_single_gpu.sh

> sh ./finetune_qwen.sh # you need to put this script under Qwen folder of Qwen source code, and adjust the dataset path folder
> python run_finetuned_qwen # run for finetuned llam3-8B model, you need to adjust the model path

### Experimental Results

For more details and results, please checkout 
[datasets and experiement results details](DS_ER_README.md)

> S: with few shot strategy; F: with fine-tuned strategy

#### COVID-19-RELATED TWEETS Sentiment classification results


| Model       | ACC($\uparrow$) | F1($\uparrow$) | U/E($\downarrow$) |
|-------------|-----------------|----------------|-------------------|
| Gemini-pro(S)  | 0.4888(-0.014) | 0.4880(-0.022) | 0.0375(-0.001)    |
| Llama-3-8B(S)  | 0.5363(+0.025) | 0.5298(+0.015) | 0.0000(-0.001)    |
| Qwen-7B(S)     | 0.3900(-0.101) | 0.3519(-0.117) | 0.0150(+0.012)    |
| Qwen-14B(S)    | 0.4575(+0.001) | 0.4556(-0.001) | 0.0037(-0.006)    |
| Vicuna-7B(S)   | 0.3700(+0.010) | 0.3362(-0.004) | 0.0013(+0.001)    |
| Vicuna-13B(S)  | 0.5050(+0.000) | 0.4951(+0.000) | 0.0000(-0.001)    |
|-------------|-----------------|----------------|-------------------|
| Llama-3-8B(F)  | 0.4675(-0.044) | 0.4910(-0.024) | 0.1175(+0.116)    |
| Qwen-7B(F)     | **0.8388(+0.348)** | **0.8433(+0.374)** | 0.0000(+0.000)    |


#### E-Commercial Product Text Classification Results

| Model       | ACC($\uparrow$) | F1($\uparrow$) | U/E($\downarrow$) |
|-------------|-----------------|----------------|-------------------|
| Gemini-pro(S)  | 0.8862(+0.009)   | 0.8963(+0.009)   | 0.0100(+0.000)   |
| Llama-3-8B(S)  | 0.9062(-0.005)   | 0.9065(-0.005)   | 0.0000(+0.000)   |
| Qwen-7B(S)     | 0.6737(+0.089)   | 0.8226(+0.164)   | 0.1812(-0.004)   |
| Qwen-14B(S)    | 0.7887(+0.131)   | 0.8548(+0.170)   | 0.0775(-0.003)   |
| Vicuna-7B(S)   | 0.7925(+0.083)   | 0.7899(+0.074)   | 0.0000(-0.005)   |
| Vicuna-13B(S)  | 0.9075(+0.071)   | 0.9153(+0.065)   | 0.0088(-0.005)   |
|-------------|-----------------|----------------|-------------------|
| Llama-3-8B(F)  | 0.9175(+0.006)   | 0.9164(+0.003)   | 0.0000(+0.000)   |
| Qwen-7B(F)     | **0.9713(+0.386)** | **0.9713(+0.313)**   | 0.0000(-0.185)   |


#### ECONOMIC TEXTS Sentiment Classification Results

| Model       | ACC($\uparrow$) | F1($\uparrow$) | U/E($\downarrow$) |
|-------------|-----------------|----------------|-------------------|
| Gemini-pro(S)  | 0.6925(-0.056)   | 0.7217(-0.030)   | 0.0400(+0.039)   |
| Llama-3-8B(S)  | 0.7550(-0.012)   | 0.7585(-0.013)   | 0.0013(+0.000)   |
| Qwen-7B(S)     | 0.6837(-0.071)   | 0.6900(-0.069)   | 0.0288(+0.026)   |
| Qwen-14B(S)    | 0.7738(-0.011)   | 0.7748(-0.011)   | 0.0063(+0.001)   |
| Vicuna-7B(S)   | 0.7738(+0.031)   | 0.7607(+0.036)   | 0.0000(+0.000)   |
| Vicuna-13B(S)  | 0.7575(+0.082)   | 0.7616(+0.088)   | 0.0013(+0.000)   |
|-------------|-----------------|----------------|-------------------|
| Llama-3-8B    | 0.7913(+0.024)   | 0.7796(+0.009)   | 0.0000(-0.001)   |
| Qwen-7B(F)    | **0.8400(+0.085)** | **0.8302(+0.074)** | 0.0000(-0.003)   |


#### SMS SPAM COLLECTION Classification Results

| Model       | ACC($\uparrow$) | F1($\uparrow$) | U/E($\downarrow$) |
|-------------|-----------------|----------------|-------------------|
| Gemini-pro(S)  | 0.8163(+0.166)   | 0.8759(+0.136)   | 0.0488(-0.009)   |
| Llama-3-8B(S)  | 0.5825(+0.189)   | 0.6482(+0.206)   | 0.0088(+0.006)   |
| Qwen-7B(S)     | 0.7525(+0.047)   | 0.8124(+0.060)   | 0.0362(+0.035)   |
| Qwen-14B(S)    | 0.8525(-0.061)   | 0.8730(-0.048)   | 0.0025(+0.003)   |
| Vicuna-7B(S)   | 0.5675(+0.291)   | 0.6310(+0.346)   | 0.0013(+0.001)   |
| Vicuna-13B(S)  | 0.6412(+0.186)   | 0.6976(+0.183)   | 0.0000(+0.000)   |
|-------------|-----------------|----------------|-------------------|
| Llama-3-8B(F)  | 0.9825(+0.589)   | 0.9826(+0.540)   | 0.0000(-0.003)   |
| Qwen-7B(F)     | **0.9938(+0.289)** | **0.9937(+0.241)** | 0.0000(+0.000)   |


## Cite
If you can find our project is useful, please cite our paper

[Smart Expert System: Large Language Models as Text Classifiers](https://arxiv.org/abs/2405.10523):

```
@article{wang2024smart,
  title={Smart Expert System: Large Language Models as Text Classifiers},
  author={Wang, Zhiqiang and Pang, Yiran and Lin, Yanbin},
  journal={arXiv preprint arXiv:2405.10523},
  year={2024}
}
```

[Large Language Models Are Zero-Shot Text Classifiers](https://arxiv.org/abs/2312.01044):
```
@article{wang2023large,
  title={Large Language Models Are Zero-Shot Text Classifiers},
  author={Wang, Zhiqiang and Pang, Yiran and Lin, Yanbin},
  journal={arXiv preprint arXiv:2312.01044},
  year={2023}
}
```
