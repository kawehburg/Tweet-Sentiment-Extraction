# Tweet Sentiment Extraction

这是Tweet Sentiment Extraction项目

程序入口是agent.py

参数包括

- --seed：随机种子
- --data：数据集
- --model：基础模型
- --pretrained：预训练地址
- --head：HEAD模型
- --loss：loss设置
- --lr：学习率
- --schedule：学习率策略
- --train：是否在训练

训练方法


```
python agent.py --seed=1024 --data='data/extended_folds.csv' --model='roberta' --pretrained='roberta' --head='linear' --loss='ce' --lr=3e-5 --train=true
```

