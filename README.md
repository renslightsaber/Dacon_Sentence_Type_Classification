
# [Dacon 문장 유형 분류 AI 경진대회](https://dacon.io/competitions/official/236037/overview/description)
<img src="/img/스크린샷 2023-03-11 오후 2.46.46.png" width="99%"></img>
## Competition Info
 - Period: 2022.12.12 - 2022.12.23
 - Joined as: Team with [A.RYANG](https://github.com/nomaday)
 - TEAM_NAME: '활기력'
 - TASK: `Multi-Label Classification`
 - Evaluation Metric: `Weighted F1 Score`
 - Environment: Colab 


## Result
 - PUBLIC  : 0.75495  |  16th / 335 
 - PRIVATE : 0.75144  |  22nd / 333 (Top 7%)  


## What I learned from this Competition:
 - How to write code and build model for `Multi-Label Classification`.

## Tried to solve 'Class Imbalanced' problem.
 There was severe Imbalanced per each label. Therefore, we researched how to solve this and found thses methods.
 
  - Sampler
  - Class Weights -> Loss Function
  - Build and train(=fine-tune) Model per each label 

 However, none of these work better than codes on github. Instead of trying to solve this, We concentrated on build better Fine-tuning Model (`tunib/electra-ko-en-base` or add `nn.BatchNorm2d()` or `nn.LeakyReLU()` Layer) and other methods(such as `clean_text`).


### How to train or inference in CLI? [Check Here](https://github.com/renslightsaber/Dacon_Sentence_Type_Classification/blob/main/how_to_train_inference.md)
 
 
