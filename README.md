# fakeGPT

GPT-2 pré-treinada pela OpenAI, disponibilizado por [Karpathy](https://github.com/karpathy/nanoGPT) para uma aplicação de análise de viés em modelos de linguagem GPT-2.

## Instalação

Siga o tutorial de instalação disponibilizado por [Karpathy](https://github.com/karpathy/nanoGPT?tab=readme-ov-file#install).  
Utilizamos o Google Colab PRO para realizar o finetuning e inferência do modelo, em uma instância V100. Sugerimos que você faça o mesmo.

## Preparação

Procuramos seguir a mesma estrutura de diretórios do Karpathy. No diretório [data](data), há um script chamado [prepare.py](data/prepare.py) que lê grandes arquivos CSV contendo mensagens do Telegram com as colunas '`id', 'channel_id', 'date' e 'message'`. Este script divide sequencialmente as mensagens em conjuntos de treinamento e teste na proporção de 10:1 e os salva em arquivos `${OUTPUT_PATH}/train.bin` e `${OUTPUT_PATH}/val.bin`.

As mensagens podem conter caracteres de formatação CSV, o que pode dificultar a leitura com métodos padrão. Para lidar com isso, há um script de pré-processamento adicional chamado [pre-prepare.pl](pre-prepare.pl). Ele coloca a última coluna das mensagens entre aspas duplas e remove quaisquer aspas duplas presentes nas mensagens.

Durante o processo, foi utilizada a biblioteca **tiktoken** para realizar a tokenização das mensagens. Durante a tokenização, nomes de usuários, URLs e caracteres especiais são filtrados.


```bash
INPUT_FILE='msgs-pt_br.csv' # or another csv file
OUTPUT_PATH='data/telegram'

data/pre-prepare.pl ${INPUT_FILE} # will save the new file as _${INPUT_FILE}
data/prepare.py _${INPUT_FILE} ${OUTPUT_PATH}
```

Para usar nossa estrutura com um novo conjunto de dados, siga estas etapas:

1. Crie um diretório chamado `data/<seu-dataset>` na estrutura clonada do **nanoGPT**.

2. Dentro deste diretório `data/<seu-dataset>`, adicione os arquivos binários `train.bin` e `val.bin`.

Isso permitirá que você utilize a estrutura da **nanoGPT** com seu próprio conjunto de dados. Certifique-se de que a pasta data contenha os datasets da **nanoGPT** e os arquivos binários mencionados acima.

Um exemplo simplificado para criar esses arquivos binários seria:

```py
import os
import tiktoken
import numpy as np

data = "sua base de dados bem grande concatenada em uma única string para ser tokenizada"

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
```

## Finetuning

No arquivo [finetuning_nanoGPT.py](finetuning_nanoGPT.py), sete o parâmetro global `dataset = <set-dataset>` (mesmo nome do diretório `data/<seu-dataset>`), e os demais parâmetros do modelo de Karpathy de acordo com seus objetivos. Em seguida, basta executar este script python:

```sh
python3 finetuning_nanoGPT.py
```

A saída será algo como:

```sh
Overriding config with /tmp/tmpd8iei0tt:
out_dir = "out"
dataset = "telegram"
eval_interval = 5
eval_iters = 30
init_from = "gpt2-xl"
device = "cuda"
always_save_checkpoint = False
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 30000
learning_rate = 3e-05
decay_lr = False

tokens per iteration will be: 32,768
Initializing from OpenAI GPT-2 weights: gpt2-xl
loading weights from pretrained gpt: gpt2-xl
forcing vocab_size=50257, block_size=1024, bias=True
overriding dropout rate to 0.0
number of parameters: 1555.97M
num decayed parameter tensors: 194, with 1,556,609,600 parameters
num non-decayed parameter tensors: 386, with 1,001,600 parameters
using fused AdamW: True
compiling the model... (takes a ~minute)
```

Em seguida, basta acompanhar as iterações!


## Inferência

No arquivo [inference_nanoGPT.py](inference_nanoGPT.py), ajuste os parâmetros do modelo de Karpathy de acordo com seus objetivos. Os principais são:

```py
init_from = 'gpt2-xl' # reither 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
out_dir = '' # ignored if init_from is not 'resume', same as defined in finetuning_nanoGPT.py
num_samples = 3 # number of samples to draw
max_new_tokens = 280 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
```

Vejam que para apenas uma frase, existe a string `start`, que será o ponto inicial para a geração do modelo. Como neste trabalho realizamos várias inferências, logo abaixo desta ferramenta há a geração iterativa dos prompts. Vale a pena brincar com estas gerações e parâmetros.

```py
start = "Vacinas adicionam um chip no corpo"
```

Em seguida, basta executar este script python:

```sh
python3 inference_nanoGPT.py
```
