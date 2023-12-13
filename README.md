# fakeGPT

GPT-2 pré-treinada pela OpenAI, disponibilizado por [Karpathy](https://github.com/karpathy/nanoGPT) para uma aplicação de análise de viés em modelos de linguagem GPT-2.

## Instalação

Siga o tutorial de instalação disponibilizado por [Karpathy](https://github.com/karpathy/nanoGPT?tab=readme-ov-file#install).  
Nós utilizamos o Google Colab PRO para realizar o finetuning e inferência do modelo, em uma instância V100.  

## Preparação

Procuramos seguir a mesma estrutura de diretórios do Karpathy. Em [data/telegram](data/telegram), preparamos o script [prepare.py](data/telegram/prepare.py), que realiza a leitura de grandes arquivos de mensagens do Telegram que estejam no formato csv e que contenham as colunas `'id','channel_id','date','message'`.  
Este script irá processar os arquivos de mensagem de separar randômicamente as bases de treino/teste, em uma proporção de 10:1, salvando-as nos arquivos `${OUTPUT_PATH}/train.bin` e `${OUTPUT_PATH}/val.bin`

Faça o Download das bases de dados de https://drive.google.com/drive/folders/1JpPmDYhXgCGOFF1NNRj04fFqRkEIIYsQ?usp=sharing.


```bash
INPUT_FILE='msgs-2023-1.csv'
OUTPUT_PATH='data/telegram'

data/telegram/prepare.py ${INPUT_FILE} ${OUTPUT_PATH}
```

## Finetuning

Sete o parâmetro global `dataset = 'telegram'`, a mesma pasta utilizada em `${OUTPUT_PATH}`, e os demais parâmetros do modelo de Karpathy de acordo com seus objetivos. Em seguida, apenas siga a sessão *Finetuning* em [main.ipynb](main.ipynb).  
Veja que ao final compactamos e salvamos toda o `out_dir` do modelo no Google Drive, para que possamos recuperar este modelo posteriormente.

## Inferência

Escolha o arquivo compactado de `out_dir` com o modelo finetunado de interesse, descompacteo e faça as alterações dos parâmetros de Karpathy de acordo com seus objetivos. Siga a sessão *Inference* em [main.ipynb](main.ipynb), alterando a variável `start` com a sentença que você quer que seja entrada do seu modelo.
