import math
import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Valores de var_smoothing testados na busca rápida de hiperparâmetros (tune).
VAR_SMOOTHING_GRID = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Busca profunda (deep_tune): grade fina (~100 valores) e priors uniformes vs aprendidos,
# avaliados com validação cruzada estratificada 10-fold repetida 5x (mais lento, robusto).
DEEP_VAR_SMOOTHING_GRID = np.logspace(-12, 0, 200).tolist()

# Decaimento por recência: jogos mais antigos recebem peso menor.
# RECENCY_HALF_LIFE é em anos — um jogo com essa idade vale metade de um jogo atual.
# 0 = sem decaimento (todos os jogos com peso igual, comportamento original).
# Configurável também via env var WORLDCUP_HALF_LIFE.
RECENCY_HALF_LIFE = float(os.environ.get("WORLDCUP_HALF_LIFE", "4"))

# Configuração do chute de placar (Poisson). Ataque = gols marcados; defesa = gols sofridos.
SCORE_MAX_GOALS = 10          # teto de gols considerado ao montar o placar
SCORE_ATTACK_WEIGHT = 0.6    # peso do ataque do time vs defesa do adversário (0..1)
SCORE_HOME_ADVANTAGE = 1.1   # fator de mando: >1 dá um leve boost de gols ao mandante


def _build_classifier():
    # As features têm escalas muito diferentes (taxa 0-1 vs contagens de confronto direto);
    # o StandardScaler normaliza tudo antes do Naive Bayes, melhorando o treinamento.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("nb", GaussianNB()),
    ])

LABELS = ["home team win", "draw", "away team win"]

# Pós-processamento das probabilidades para que a confiança seja mais realista:
# o Naive Bayes tende a saturar perto de 100%, então amaciamos a distribuição
# (temperatura) e limitamos cada classe a um teto/piso. Nenhum resultado vira certeza.
TEMPERATURE = 1.5   # T>1 achata a distribuição; quanto maior, menos confiante
MAX_PROB = 0.95     # teto: nenhuma classe passa de 95%
MIN_PROB = 0.01     # piso: nenhuma classe fica abaixo de 1%

# Boost de empate: multiplica a probabilidade de empate antes de renormalizar.
# Compensa o sub-peso do empate nos dados históricos de Copa (apenas ~15% dos jogos).
# 1.0 = sem boost. Configurável via WORLDCUP_DRAW_BOOST.
DRAW_BOOST = float(os.environ.get("WORLDCUP_DRAW_BOOST", "2.2"))

# Threshold de empate: se a margem entre o favorito e o empate for menor que este valor
# (em probabilidade), declara empate em vez de vitória.
# 0.0 = desativado (usa apenas o argmax). Configurável via WORLDCUP_DRAW_THRESHOLD.
DRAW_THRESHOLD = float(os.environ.get("WORLDCUP_DRAW_THRESHOLD", "0.08"))

# As 10 features numéricas que descrevem um confronto, na ordem usada pelo classificador:
FEATURE_NAMES = [
    "home_win_rate",         # taxa de vitórias do mandante em toda a sua história
    "away_win_rate",         # taxa de vitórias do visitante em toda a sua história
    "home_avg_goals_for",    # média de gols marcados pelo mandante por jogo
    "home_avg_goals_against",# média de gols sofridos pelo mandante por jogo
    "away_avg_goals_for",    # média de gols marcados pelo visitante por jogo
    "away_avg_goals_against",# média de gols sofridos pelo visitante por jogo
    "h2h_home_wins",         # confronto direto: vitórias do mandante sobre o visitante
    "h2h_draws",             # confronto direto: empates entre os dois times
    "h2h_away_wins",         # confronto direto: vitórias do visitante sobre o mandante
    "home_advantage",        # fator de mando de campo (fixo, sinaliza quem joga em casa)
]


class Model:
    def __init__(self, repository):
        self.repo = repository
        self.clf = _build_classifier()
        self._matches = None
        self._team_stats = None
        self._h2h = None

    def _load(self):
        if self._matches is None:
            self._matches = self.repo.get_matches()
            appearances = self.repo.get_team_appearances()
            self._team_stats = self._compute_team_stats(appearances)
            self._h2h = self._compute_h2h(self._matches)

    @staticmethod
    def _recency_weight(year):
        # Peso exponencial: w = 2^(-(current_year - year) / half_life).
        # half_life=0 desativa o decaimento (peso 1 para todos).
        if not RECENCY_HALF_LIFE:
            return 1.0
        current_year = 2026
        age = max(current_year - year, 0)
        return math.pow(2.0, -age / RECENCY_HALF_LIFE)

    @staticmethod
    def _compute_team_stats(appearances):
        # Resume a história de cada time ponderando cada jogo por recência.
        # As contagens ficam em "peso acumulado" (floats), e as taxas são calculadas
        # depois dividindo pelo peso total — equivale a média ponderada.
        stats = {}
        for a in appearances:
            w = Model._recency_weight(a["year"])
            s = stats.setdefault(
                a["team_id"],
                {"played": 0.0, "wins": 0.0, "goals_for": 0.0, "goals_against": 0.0},
            )
            s["played"] += w
            s["wins"] += a["win"] * w
            s["goals_for"] += a["goals_for"] * w
            s["goals_against"] += a["goals_against"] * w
        return stats

    @staticmethod
    def _compute_h2h(matches):
        # Confronto direto (head-to-head) por par de times, guardado pela ótica de cada time.
        h2h = {}
        for m in matches:
            home, away = m["home_team_id"], m["away_team_id"]
            key = frozenset((home, away))
            rec = h2h.setdefault(key, {})
            pair = rec.setdefault(home, {"wins": 0, "draws": 0, "losses": 0})
            opp = rec.setdefault(away, {"wins": 0, "draws": 0, "losses": 0})
            if m["result"] == "home team win":
                pair["wins"] += 1
                opp["losses"] += 1
            elif m["result"] == "away team win":
                pair["losses"] += 1
                opp["wins"] += 1
            else:
                pair["draws"] += 1
                opp["draws"] += 1
        return h2h

    def _team_feature(self, team_id):
        s = self._team_stats.get(team_id)
        if not s or s["played"] == 0:
            return 0.0, 0.0, 0.0
        return (
            s["wins"] / s["played"],
            s["goals_for"] / s["played"],
            s["goals_against"] / s["played"],
        )

    def _h2h_feature(self, home_id, away_id):
        rec = self._h2h.get(frozenset((home_id, away_id)))
        if not rec or home_id not in rec:
            return 0, 0, 0
        r = rec[home_id]
        return r["wins"], r["draws"], r["losses"]

    def _features(self, home_id, away_id, home_advantage=1.0):
        # Monta uma linha com as 10 features numéricas (ver FEATURE_NAMES) para o GaussianNB.
        h_wr, h_gf, h_ga = self._team_feature(home_id)
        a_wr, a_gf, a_ga = self._team_feature(away_id)
        h2h_hw, h2h_d, h2h_aw = self._h2h_feature(home_id, away_id)
        return [
            h_wr,
            a_wr,
            h_gf,
            h_ga,
            a_gf,
            a_ga,
            float(h2h_hw),
            float(h2h_d),
            float(h2h_aw),
            home_advantage,
        ]

    def _build_dataset(self):
        X, y = [], []
        for m in self._matches:
            X.append(self._features(m["home_team_id"], m["away_team_id"]))
            y.append(m["result"])
        return np.array(X, dtype=float), np.array(y)

    def train(self):
        self._load()
        X, y = self._build_dataset()
        self.clf.fit(X, y)
        return self

    def tune(self, cv=5):
        # Busca o melhor var_smoothing por validação cruzada e adota o melhor modelo
        # (o GridSearchCV já re-treina no conjunto inteiro). Retorna parâmetro e acurácia.
        self._load()
        X, y = self._build_dataset()
        search = GridSearchCV(
            _build_classifier(),
            {"nb__var_smoothing": VAR_SMOOTHING_GRID},
            cv=cv,
        )
        search.fit(X, y)
        self.clf = search.best_estimator_
        return {
            "var_smoothing": search.best_params_["nb__var_smoothing"],
            "accuracy": float(search.best_score_),
        }

    def deep_tune(self, progress_callback=None, n_jobs=-1):
        # Busca profunda: grade fina de var_smoothing + priors uniformes vs aprendidos,
        # com CV estratificada 10-fold repetida 5x. As combinações são avaliadas em
        # paralelo (joblib, n_jobs=-1 usa todos os núcleos); a CV interna fica serial
        # para não aninhar paralelismo. progress_callback(feitos, total) anima o progresso.
        self._load()
        X, y = self._build_dataset()
        n_classes = len(np.unique(y))
        uniform = [1.0 / n_classes] * n_classes
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)

        combos = [(vs, p) for p in (None, uniform) for vs in DEEP_VAR_SMOOTHING_GRID]
        total = len(combos)

        def score_combo(var_smoothing, priors):
            clf = _build_classifier()
            clf.set_params(nb__var_smoothing=var_smoothing, nb__priors=priors)
            return cross_val_score(clf, X, y, cv=cv, n_jobs=1).mean()

        # return_as="generator" entrega os resultados conforme terminam, permitindo
        # atualizar a barra de progresso durante a execução paralela.
        results = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(score_combo)(vs, p) for vs, p in combos
        )

        best = {"accuracy": -1.0}
        for i, (score, (var_smoothing, priors)) in enumerate(zip(results, combos), start=1):
            if score > best["accuracy"]:
                best = {
                    "accuracy": float(score),
                    "var_smoothing": var_smoothing,
                    "priors": "uniforme" if priors is not None else "aprendido",
                    "_estimator_params": (var_smoothing, priors),
                }
            if progress_callback:
                progress_callback(i, total)

        var_smoothing, priors = best.pop("_estimator_params")
        self.clf = _build_classifier()
        self.clf.set_params(nb__var_smoothing=var_smoothing, nb__priors=priors)
        self.clf.fit(X, y)
        return best

    def strength_scores(self, home_id, away_id):
        """Heurística de força relativa: taxa de vitórias mais o saldo médio de gols."""
        h_wr, h_gf, h_ga = self._team_feature(home_id)
        a_wr, a_gf, a_ga = self._team_feature(away_id)
        home = h_wr + 0.1 * (h_gf - h_ga)
        away = a_wr + 0.1 * (a_gf - a_ga)
        return {"home": round(home, 4), "away": round(away, 4)}

    def predict_score(self, home_id, away_id, outcome=None, max_goals=SCORE_MAX_GOALS):
        # Quantidade de gols de cada time = média esperada (lambda), vinda do seu ataque
        # (gols marcados) combinado com a defesa (gols sofridos) do adversário, ponderada
        # por SCORE_ATTACK_WEIGHT e com o mando de campo. O placar é o lambda arredondado
        # ao inteiro (reflete a média real, sem o viés de placar baixo da moda da Poisson).
        # A Poisson é usada para ajustar a diferença de gols quando o placar arredondado
        # contradiz o resultado previsto (outcome).
        _, h_gf, h_ga = self._team_feature(home_id)
        _, a_gf, a_ga = self._team_feature(away_id)
        w = SCORE_ATTACK_WEIGHT
        lam_home = max(w * h_gf + (1 - w) * a_ga, 1e-6) * SCORE_HOME_ADVANTAGE
        lam_away = max(w * a_gf + (1 - w) * h_ga, 1e-6)

        home_goals = round(lam_home)
        away_goals = round(lam_away)

        if outcome and not self._matches_outcome(home_goals, away_goals, outcome):
            home_goals, away_goals = self._fit_outcome(lam_home, lam_away, outcome, max_goals)

        return {
            "home_goals": home_goals,
            "away_goals": away_goals,
            "expected": {"home": round(lam_home, 2), "away": round(lam_away, 2)},
        }

    @staticmethod
    def _matches_outcome(i, j, outcome):
        if outcome == "home team win":
            return i > j
        if outcome == "away team win":
            return j > i
        if outcome == "draw":
            return i == j
        return True

    def _fit_outcome(self, lam_home, lam_away, outcome, max_goals):
        # Entre os placares coerentes com o resultado previsto, escolhe o mais provável
        # segundo a Poisson (mantém a diferença de gols realista).
        p_home = [self._poisson_pmf(k, lam_home) for k in range(max_goals + 1)]
        p_away = [self._poisson_pmf(k, lam_away) for k in range(max_goals + 1)]
        candidates = [
            (i, j)
            for i in range(max_goals + 1)
            for j in range(max_goals + 1)
            if self._matches_outcome(i, j, outcome)
        ]
        return max(candidates, key=lambda ij: p_home[ij[0]] * p_away[ij[1]])

    @staticmethod
    def _poisson_pmf(k, lam):
        return math.exp(-lam) * lam ** k / math.factorial(k)

    def predict(self, home_id, away_id):
        # Pede ao classificador a probabilidade de cada resultado; em caso de empate
        # entre dois resultados, a heurística de força relativa decide qual prevalece.
        self._load()
        x = np.array([self._features(home_id, away_id)], dtype=float)
        classes = list(self.clf.classes_)
        proba = self._soften(self.clf.predict_proba(x)[0], classes)
        probabilities = {label: float(proba[classes.index(label)]) for label in LABELS}

        ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        top_prob = ranked[0][1]
        tied = [label for label, p in ranked if abs(p - top_prob) < 1e-9]

        strength = self.strength_scores(home_id, away_id)
        if len(tied) > 1:
            predicted = self._break_tie(tied, strength)
        else:
            predicted = ranked[0][0]

        # Draw threshold: se o favorito vence por margem pequena sobre o empate,
        # prefere empate (jogos equilibrados tendem a terminar empatados).
        if predicted != "draw" and DRAW_THRESHOLD > 0:
            draw_prob = probabilities.get("draw", 0.0)
            if top_prob - draw_prob < DRAW_THRESHOLD:
                predicted = "draw"

        return {
            "predicted": predicted,
            "probabilities": probabilities,
            "strength": strength,
            "score": self.predict_score(home_id, away_id, outcome=predicted),
        }

    def _soften(self, proba, classes):
        # 1) Temperatura: eleva a 1/T e renormaliza, achatando picos perto de 100%.
        p = np.power(proba, 1.0 / TEMPERATURE)
        p = p / p.sum()
        # 2) Boost de empate: reequilibra a classe sub-representada no treino.
        if DRAW_BOOST != 1.0 and "draw" in classes:
            draw_idx = list(classes).index("draw")
            p[draw_idx] *= DRAW_BOOST
            p = p / p.sum()
        # 3) Teto/piso: limita cada classe e renormaliza para voltar a somar 1.
        p = np.clip(p, MIN_PROB, MAX_PROB)
        return p / p.sum()

    @staticmethod
    def _break_tie(tied, strength):
        if strength["home"] > strength["away"] and "home team win" in tied:
            return "home team win"
        if strength["away"] > strength["home"] and "away team win" in tied:
            return "away team win"
        return tied[0]

    def evaluate(self, cv=5):
        self._load()
        X, y = self._build_dataset()
        scores = cross_val_score(_build_classifier(), X, y, cv=cv)
        return float(scores.mean())
