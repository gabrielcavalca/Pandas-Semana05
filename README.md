# Spam Classifier — Machine Learning em Python

Este projeto implementa um classificador de mensagens SMS (spam x ham) utilizando técnicas de Processamento de Linguagem Natural (NLP) e algoritmos de Machine Learning. Dois modelos foram treinados e comparados: Logistic Regression e Random Forest.

## Dataset

O projeto utiliza o dataset público SMS Spam Collection.

O dataset contém duas classes:
- ham: mensagem legítima  
- spam: mensagem indesejada  

## Pré-processamento

As seguintes etapas foram aplicadas:

1. Remoção de duplicatas.  
2. Remoção de colunas com valores nulos.  
3. Conversão dos rótulos:  
   - ham → 0  
   - spam → 1  
4. Divisão dos dados em treino e teste (80% / 20%).  
5. Vetorização das mensagens utilizando TF-IDF com stopwords em inglês.  

## Modelos Utilizados

### Logistic Regression
- Solver: liblinear  
- max_iter: 1000  
- random_state: 42  

### Random Forest
- n_estimators: 1000  
- random_state: 42  

## Resultados

### Logistic Regression
| Métrica     | Valor   |
|-------------|---------|
| Precision   | 0.9821  |
| Recall      | 0.7586  |
| Accuracy    | 0.9642  |

### Random Forest
| Métrica     | Valor   |
|-------------|---------|
| Precision   | 1.0000  |
| Recall      | 0.8069  |
| Accuracy    | 0.9729  |

## Conclusões

- Ambos os modelos obtiveram alta acurácia.
- O Random Forest obteve maior recall, identificando melhor mensagens de spam.
- Logistic Regression apresentou performance excelente com menor complexidade.
- O TF-IDF foi essencial para transformar texto em vetores numéricos.

## Como Executar

1. Clone o repositório:

```bash
git clone  https://github.com/gabrielcavalca/Pandas-Semana05.git   
cd Pandas-Semana05
```


2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o script:

```
python project.py
```


