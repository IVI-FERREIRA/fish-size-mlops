# 🐟 API de Predição de Peso de Peixes

API para estimar o **peso de um peixe (em gramas)** a partir de:

- Medidas morfométricas manuais (JSON)
- Imagem do peixe com régua de referência

Projeto desenvolvido como **teste técnico**, seguindo boas práticas de mercado, com foco em:

- Simplicidade
- Reprodutibilidade
- Separação clara entre **treinamento** e **inferência**
- Containerização
- Decisões arquiteturais conscientes

---

## 📌 Visão Geral da Arquitetura
<img width="91" height="31" alt="image" src="https://github.com/user-attachments/assets/9b0e599d-55d1-4f48-a520-766b4c7364b9" />


### 🏋️‍♂️ Treinamento
- Executado **localmente**, fora do container
- Gera o modelo final `model.pkl`
- MLflow utilizado **apenas durante o treino** (tracking experimental)

### 🚀 Inferência
- API FastAPI
- Modelo carregado via `joblib`
- **Sem uso de MLflow em runtime**
- Execução local ou via Docker

📌 O uso de MLflow foi propositalmente restrito ao treinamento para reduzir complexidade e dependências em produção.

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
```

---

## 🧠 Sobre o Modelo

Modelo de **regressão supervisionada** treinado com as seguintes features:

| Feature | Descrição |
|---------|-----------|
| Length1 | Comprimento parcial do peixe |
| Length2 | Comprimento intermediário |
| Length3 | **Comprimento total (focinho → ponta da cauda)** |
| Height  | Altura do peixe |
| Width   | Largura do peixe |

📌 **Length3 representa o tamanho total do peixe.**

---

## 🏋️‍♂️ Executando o Treinamento (Local)

### 1️⃣ Criar ambiente virtual
```bash
python -m venv .venv
```

### 2️⃣ Ativar o ambiente

**Windows**
```bash
.venv\Scripts\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 3️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 4️⃣ Executar o treino
```bash
python -m src.train
```

Ao final do treino será gerado o arquivo:

```text
model.pkl
```

📌 Este arquivo é versionado no repositório e utilizado diretamente pela API.

---

## 🚀 Executando a API Localmente (Sem Docker)

Com o ambiente virtual ativado:

```bash
uvicorn api.main:app --reload
```

A API ficará disponível em:

```text
http://localhost:8000
```

Documentação Swagger:

```text
http://localhost:8000/docs
```

---

## 🔌 Endpoints

### 🔹 POST /predict — Medidas Manuais

**Entrada**
```json
{
  "Length1": 20,
  "Length2": 22,
  "Length3": 25,
  "Height": 5,
  "Width": 5
}
```

**Resposta**
```json
{
  "estimated_weight_g": 183.36
}
```

---

### 🔹 POST /predict-image — Imagem do Peixe

Envie uma imagem contendo o peixe e uma régua de referência.

A API realiza:
- Extração de contornos com OpenCV
- Conversão de pixels → centímetros
- Geração das features morfométricas
- Estimativa do peso

**Resposta**
```json
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
```

---

## 🐳 Executando com Docker

### 1️⃣ Build da imagem
```bash
docker build --no-cache -t fish-size-mlops .
```

### 2️⃣ Rodar o container
```bash
docker run -p 8000:8000 fish-size-mlops
```

A API ficará disponível em:

```text
http://localhost:8000
```

---

## 📦 Dependências Principais

```text
pandas
numpy
scikit-learn
joblib
fastapi
uvicorn
opencv-python
python-multipart
```

### ❌ Por que o MLflow não é utilizado em runtime na API?

O MLflow foi utilizado **exclusivamente durante o treinamento**, com os seguintes objetivos:

- Tracking de métricas (MAE, R²)
- Comparação de experimentos
- Versionamento experimental de modelos

Na camada de inferência (API), o MLflow **não é utilizado propositalmente**, pelos seguintes motivos:

- Evita dependência de backend de tracking em runtime
- Reduz tempo de inicialização da API
- Simplifica a imagem Docker
- Elimina acoplamento entre API e infraestrutura de experimentos
- Facilita testes técnicos e execução local

O modelo final é **congelado** e exportado como `model.pkl`, sendo carregado diretamente via `joblib` na API.

📌 Em ambientes de produção, o MLflow pode ser integrado via **CI/CD** ou **Model Registry externo**, mas não diretamente dentro da aplicação de inferência.


---

## 🧠 Próximos Passos (Produção)

- Docker Compose (API + serviços auxiliares)
- Model Registry externo
- CI/CD para retreino automático
- Monitoramento de drift
- Visão computacional mais robusta (YOLO / segmentação)
