# 🐟 Fish Size Prediction API (MLOps Technical Test)

API para estimar o **peso de um peixe (em gramas)** a partir de:

- Medidas morfométricas manuais (JSON)
- Imagem do peixe com régua de referência

Projeto desenvolvido como **teste técnico de MLOps**, com foco em:

- Simplicidade  
- Reprodutibilidade  
- Separação clara entre **treinamento** e **inferência**  
- Containerização  
- Decisões arquiteturais conscientes  

---

## 📌 Visão Geral da Arquitetura

### Treinamento
- Executado localmente (fora do container)
- Gera o modelo final `model.pkl`

### Inferência
- API FastAPI rodando em Docker
- Carrega apenas o modelo treinado
- Sem dependências externas em runtime

### MLflow
- Utilizado **somente durante o treino**
- **Não** utilizado na API

📌 O objetivo é demonstrar boas práticas reais de MLOps em um cenário simples e funcional.

---

## 📁 Estrutura do Projeto

```text
fish-size-mlops/
│
├── api/
│   ├── __init__.py
│   └── main.py              # API FastAPI
│
├── src/
│   ├── data_prep.py         # Pré-processamento dos dados
│   ├── model.py             # Definição do modelo
│   ├── train.py             # Treinamento
│   ├── inference.py         # Inferência (model.pkl)
│   └── vision.py            # Extração de medidas da imagem
│
├── data/
│   ├── raw/                 # Dados brutos
│   └── processed/           # Dados processados
│
├── model.pkl                # Modelo final usado pela API
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md

🧠 Sobre o Modelo

Modelo de regressão supervisionada, treinado com as seguintes features:

Feature	Descrição
Length1	Comprimento parcial do peixe
Length2	Comprimento intermediário
Length3	Comprimento total (focinho → ponta da cauda)
Height	Altura do peixe
Width	Largura do peixe

📌 Length3 representa o tamanho total do peixe.

🏋️‍♂️ Treinamento do Modelo

O treinamento é executado fora do Docker.

python -m src.train


Durante o treino:

Leitura e processamento dos dados

Treinamento do modelo

Avaliação com MAE e R²

Salvamento do modelo final em:

model.pkl


Esse arquivo é:

Versionado no repositório

Copiado para dentro do container

Utilizado diretamente na inferência

❌ Por que MLflow NÃO é usado na API?

MLflow foi utilizado somente no treinamento, para:

Tracking de métricas

Experimentação

Comparação de modelos

Na API:

❌ MLflow não é utilizado

❌ mlruns não é necessário

❌ Nenhuma dependência externa em runtime

Motivos da decisão

Redução de complexidade

Menor tempo de startup

Eliminação de dependências externas

Docker mais simples

Adequado para teste técnico e produção simples

📌 Em produção real, MLflow seria utilizado via CI/CD ou Model Registry externo, não embutido na API.

🚀 Rodando a API com Docker
1️⃣ Build da imagem
docker build --no-cache -t fish-size-mlops .

2️⃣ Rodar o container
docker run -p 8000:8000 fish-size-mlops


A API ficará disponível em:

http://localhost:8000


Documentação Swagger:

http://localhost:8000/docs

🔌 Endpoints
🔹 POST /predict — Medidas manuais

Entrada (JSON):

{
  "Length1": 20,
  "Length2": 22,
  "Length3": 25,
  "Height": 5,
  "Width": 5
}


Resposta:

{
  "estimated_weight_g": 183.36
}

🔹 POST /predict-image — Imagem do peixe

Envie uma imagem contendo o peixe e uma régua de referência

A API:

Extrai contornos com OpenCV

Converte pixels → centímetros

Gera as features

Estima o peso

Exemplo com curl:

curl -X POST "http://localhost:8000/predict-image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@peixe.jpg"


Resposta:

{
  "features_extracted": {
    "Length1": 14.56,
    "Length2": 18.30,
    "Length3": 22.10,
    "Height": 4.98,
    "Width": 1.49
  },
  "estimated_weight_g": 42.54
}

📦 Dependências Principais
pandas
numpy
scikit-learn
joblib
fastapi
uvicorn
opencv-python
python-multipart


📌 MLflow não é dependência da API.

🧪 O que está sendo avaliado no teste

Separação clara entre treino e inferência

Docker funcional

API documentada

Decisões arquiteturais justificáveis

Código organizado e reproduzível

🧠 Próximos Passos (Produção)

Docker Compose (API + Registry + DB)

Model Registry externo

CI/CD para retreino automático

Monitoramento de drift

Visão computacional mais robusta (YOLO / segmentação)