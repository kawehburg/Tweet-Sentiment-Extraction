# Tweet Sentiment Extraction

这是Tweet Sentiment Extraction项目

程序入口是agent.py

参数包括

- --seed：随机种子，default=1024
- --data：数据集，default=train
- --model：基础模型，default=roberta
- --pretrained：预训练地址，default=roberta
- --head：HEAD模型，default=linear
- --loss：loss设置，default=ce
- --lr：学习率，default=3e-5
- --schedule：学习率策略，default=linear_warmup
- --batch_size：batch size，default=20
- --epochs：epochs，default=3
- --train：是否在训练，default=true

训练方法


```
python agent.py --seed=1024 --data=train --model=roberta --pretrained=roberta --head=linear --loss=ce --lr=3e-5 --schedule=linear_warmup --batch_size=20 --epochs=3 --train=true
```

