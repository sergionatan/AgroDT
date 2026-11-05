# Digital Twin para Plantação de Oliveiras

Ecossistema em Python 3.12 para simular um Gêmeo Digital com dados sintéticos de clima, solo, crescimento e faturamento.

## Estrutura

- `data_gen.py`: geração de dados mock correlacionados.
- `analysis.py`: EDA automática + treino de modelo de produtividade.
- `app.py`: dashboard Streamlit para simulação de cenários (what-if).
- `stream_sim.py`: streaming simulado de captura de sensores em JSONL.
- `data/raw/`: dataset gerado.
- `outputs/eda/`: artefatos visuais e tabelas da análise exploratória.
- `models/`: modelo treinado e métricas.

## Como executar (Windows PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Fluxo recomendado

1. Gerar dataset sintético:

```powershell
python data_gen.py
```

2. Rodar EDA e treinamento:

```powershell
python analysis.py
```

3. Subir dashboard Streamlit:

```powershell
streamlit run app.py
```

4. Simular captura contínua (streaming):

```powershell
python stream_sim.py --interval-sec 0.2 --max-events 30 --truncate
```

## Saídas esperadas

- Dataset: `data/raw/olive_digital_twin.csv`
- EDA:
  - `outputs/eda/correlation_heatmap.png`
  - `outputs/eda/summary_stats.csv`
  - `outputs/eda/outlier_counts_iqr.csv`
  - `outputs/eda/top_correlations.csv`
- Modelo:
  - `models/olive_productivity_model.joblib`
  - `models/metrics.json`
- Streaming:
  - `data/stream/olive_sensor_stream.jsonl`
