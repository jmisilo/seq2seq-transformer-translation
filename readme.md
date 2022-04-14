### Transformer Translation model

&nbsp;&nbsp;&nbsp;&nbsp;Implementation of seq2seq model with Transformers. It tries to be clear, due to possible future re-usage. Model could be used for translation tasks on different datasets. I used it for polish to english translation with data from [opus.nlpl.eu](https://opus.nlpl.eu/opus-100.php) and german to english translation using Multi30k dataset (uploaded with torchvision notebook) to show that reimplementation of this aproach is not hard. There is a few things to change.

&nbsp;&nbsp;&nbsp;&nbsp;I also created function for translation, using trained model. It could be adapt for other implementation.

&nbsp;&nbsp;&nbsp;&nbsp;Sources:
- https://arxiv.org/pdf/1706.03762.pdf - "Attention Is All You Need" paper
- https://github.com/karpathy/minGPT/blob/master/mingpt/model.py - minGPT, [Andrej Karpathy](https://karpathy.ai/) 
- https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py - Transformer from scratch, [Aladdin Persson](https://github.com/aladdinpersson)
- https://medium.com/monimoyd/step-by-step-machine-translation-using-transformer-and-multi-head-attention-96435675be75 - Step by Step Machine Translation using Transformer and Multi Head Attention, Monimoy Purkayastha