# fakeGPT

GPT-2 pré-treinada pela OpenAI, disponibilizado por [Karpathy](https://github.com/karpathy/nanoGPT) para uma aplicação de análise de viés em modelos de linguagem GPT-2.

## Instalação

Siga o tutorial de instalação disponibilizado por [Karpathy](https://github.com/karpathy/nanoGPT?tab=readme-ov-file#install).  
Utilizamos o Google Colab PRO para realizar o finetuning e inferência do modelo, em uma instância V100. Sugerimos que você faça o mesmo.  
Para remoção dos erros de ortografia, instale Enelvo: `pip3 install enelvo`.

## Preparação

Procuramos seguir a mesma estrutura de diretórios do Karpathy. No diretório [data](data), há um script chamado [prepare.py](data/prepare.py) que lê grandes arquivos CSV contendo mensagens do Telegram com as colunas '`id', 'channel_id', 'date' e 'message'`. Este script divide aleatoriamente as mensagens em conjuntos de treinamento e teste na proporção de 10:1 e os salva em arquivos `${OUTPUT_PATH}/train.bin` e `${OUTPUT_PATH}/val.bin`.

As mensagens podem conter caracteres de formatação CSV, o que pode dificultar a leitura com métodos padrão. Para lidar com isso, há um script de pré-processamento adicional chamado [pre-prepare.pl](pre-prepare.pl). Ele coloca a última coluna das mensagens entre aspas duplas e remove quaisquer aspas duplas presentes nas mensagens.

Durante o processo, foram utilizadas bibliotecas como **Enelvo** para normalizar o texto (corrigir erros de ortografia e linguagem típica da internet) e **tiktoken** para realizar a tokenização das mensagens. Durante a tokenização, nomes de usuários, URLs e caracteres especiais foram filtrados.

Você pode baixar as bases de dados em [https://drive.google.com/drive/folders/1JpPmDYhXgCGOFF1NNRj04fFqRkEIIYsQ?usp=sharing.](https://drive.google.com/drive/folders/1eMl3DENPS3-T6MV4AqqH4KJ5k7NdnDlo?usp=sharing)


```bash
INPUT_FILE='msgs-pt_br.csv'
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

Sete o parâmetro global `dataset = <set-dataset>` (mesmo do diretório `data/<seu-dataset>`), e os demais parâmetros do modelo de Karpathy de acordo com seus objetivos. Em seguida, apenas siga a sessão *Finetuning* em [main.ipynb](main.ipynb).  
Veja que ao final compactamos e salvamos todo o `out_dir` do modelo no Google Drive, para que possamos recuperar este modelo posteriormente.

## Inferência

Escolha o arquivo compactado de `out_dir` com o modelo finetunado de interesse, descompacteo e faça as alterações dos parâmetros de Karpathy de acordo com seus objetivos. Siga a sessão *Inference* em [main.ipynb](main.ipynb), alterando a variável `start` com a sentença que você quer que seja entrada do seu modelo. Esta sessão também serializa o modelo de inferência em um arquivo pikle (`<pikle_model_filename>`), que pode ser utilizado num terminal interativo.

Para executar terminal interativo de inferência [interview.py](interview.py), execute:

```bash
python interview.py <pikle_model_filename>
```

Você entra com a frase e ele realiza a inferência.
