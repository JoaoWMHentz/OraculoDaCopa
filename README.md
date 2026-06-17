# Oráculo da Copa

A utility-based agent that predicts the outcome of a World Cup match between two
national teams — **home win**, **draw**, or **away win** — from historical match data.

## How it works

- A **Naive Bayes** classifier (`GaussianNB`) estimates the probability of each outcome.
- A **relative-strength heuristic** (historical win rate plus average goal differential)
  ranks the teams and breaks ties between equally probable outcomes.
- The output is the most probable result, the probability of each of the three outcomes,
  and which team is historically stronger.

Features engineered per matchup: each team's overall win rate, average goals scored and
conceded, the head-to-head record, and a home-advantage flag.

## Architecture

```
app.py  ->  service.py  ->  model.py  ->  repository.py  ->  SQLite
```

- `repository.py` — the only file that touches the database (all SQL lives here).
- `model.py` — feature engineering, Naive Bayes training, prediction, heuristic.
- `service.py` — wires the repository into the model, validates input, shapes the response.
- `app.py` — CLI entry point.

## Data

Reads directly from the `jfjelstul/worldcup` SQLite database. Expected default path:

```
/home/joao/Documents/Repositories/Catolica/IA/worldcup/data-sqlite/worldcup.db
```

Override with the `WORLDCUP_DB` environment variable:

```bash
export WORLDCUP_DB=/path/to/worldcup.db
```

### Janela histórica

Por padrão o modelo usa apenas as Copas a partir de **1992** (cerca de 30 anos),
porque eras muito antigas distorcem a força atual dos times. Ajuste com
`WORLDCUP_MIN_YEAR` (use `0` para considerar toda a história desde 1930):

```bash
export WORLDCUP_MIN_YEAR=1992   # padrão
export WORLDCUP_MIN_YEAR=0      # sem limite
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then enter a home team and an away team by name (e.g. `Brazil`, `Argentina`).
Leave an input empty to quit.

Show the model accuracy on startup with:

```bash
python app.py --eval
```

This prints a 5-fold cross-validation accuracy number for the technical report.

Otimize os hiperparâmetros antes de treinar com:

```bash
python app.py --tune
```

O `--tune` busca o melhor `var_smoothing` do Naive Bayes por validação cruzada
(`GridSearchCV`) e adota o modelo vencedor, exibindo o parâmetro e a acurácia escolhidos.
