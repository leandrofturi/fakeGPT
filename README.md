# fakeGPT

GPT-2 pré-treinada pela OpenAI, disponibilizado por [Karpathy](https://github.com/karpathy/nanoGPT) para uma aplicação de análise de viés em modelos de linguagem GPT-2.

## Instalação

Siga o tutorial de instalação disponibilizado por [Karpathy](https://github.com/karpathy/nanoGPT?tab=readme-ov-file#install).  
Nós utilizamos o Google Colab PRO para realizar o finetuning e inferência do modelo, em uma instância V100.  

Para remoção dos erros de ortografia, instale Enelvo: `pip3 install enelvo`.

## Preparação

Procuramos seguir a mesma estrutura de diretórios do Karpathy. Em [data](data), preparamos o script [prepare.py](data/prepare.py), que realiza a leitura de grandes arquivos de mensagens em lotes do Telegram que estejam no formato csv e que contenham as colunas `'id','channel_id','date','message'`.  
Este script irá processar os arquivos de mensagem de separar randômicamente as bases de treino/teste, em uma proporção de 10:1, salvando-as nos arquivos `${OUTPUT_PATH}/train.bin` e `${OUTPUT_PATH}/val.bin`.  
As colunas de mensagens possuem os caracteres de estrutura do prórpio csv, dificultando a leitura através de métodos mais robustos. Para contornar isso, construímos mais um stript de pré-preparo: [pre-prepare.pl](data/pre-prepare.pl). Ele coloca a última coluna completamente entre aspas duplas e remove qualquer aspas duplas destas mensagens.  

Foram utilizadas bibliotecas como [Enelvo](https://github.com/thalesbertaglia/enelvo.git), para normalizar textos com características da web (como erros de ortografia e linguagem típica da internet), e [tiktoken](https://github.com/openai/tiktoken.git), uma ferramenta de tokenização da OpenAI. Durante esse estágio, foram filtrados nomes de usuários, URLs e caracteres especiais.

Faça o download das bases de dados de https://drive.google.com/drive/folders/1JpPmDYhXgCGOFF1NNRj04fFqRkEIIYsQ?usp=sharing.


```bash
INPUT_FILE='msgs-2023-1.csv'
OUTPUT_PATH='data/telegram'

data/prepare.py ${INPUT_FILE} ${OUTPUT_PATH}
```

## Finetuning

Sete o parâmetro global `dataset = 'telegram' # or 'messages'`, a mesma pasta utilizada em `${OUTPUT_PATH}`, e os demais parâmetros do modelo de Karpathy de acordo com seus objetivos. Em seguida, apenas siga a sessão *Finetuning* em [main.ipynb](main.ipynb).  
Veja que ao final compactamos e salvamos toda o `out_dir` do modelo no Google Drive, para que possamos recuperar este modelo posteriormente.

## Inferência

Escolha o arquivo compactado de `out_dir` com o modelo finetunado de interesse, descompacteo e faça as alterações dos parâmetros de Karpathy de acordo com seus objetivos. Siga a sessão *Inference* em [main.ipynb](main.ipynb), alterando a variável `start` com a sentença que você quer que seja entrada do seu modelo.
