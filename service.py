from difflib import get_close_matches

from model import Model
from repository import Repository


class Service:
    """Thin layer between the app and the model."""

    def __init__(self, repository=None):
        self.repo = repository or Repository()
        self.model = Model(self.repo)
        self._trained = False
        self._names = None

    def is_known(self, name):
        return self.repo.resolve_team_name(name) is not None

    def suggest(self, name, limit=3):
        # Sugere nomes parecidos (erros de digitação) por similaridade de texto.
        if self._names is None:
            self._names = self.repo.list_team_names()
        lower_to_name = {n.lower(): n for n in self._names}
        matches = get_close_matches(name.lower(), lower_to_name.keys(), n=limit, cutoff=0.6)
        return [lower_to_name[m] for m in matches]

    def setup(self, tune=False, deep=False, progress_callback=None):
        # tune busca hiperparâmetros antes de fixar o modelo; deep faz a busca profunda.
        if deep:
            result = self.model.deep_tune(progress_callback=progress_callback)
        elif tune:
            result = self.model.tune()
        else:
            result = self.model.train()
        self._trained = True
        return result

    def list_team_names(self):
        return self.repo.list_team_names()

    def evaluate(self, cv=5):
        return self.model.evaluate(cv=cv)

    def predict(self, home_name, away_name):
        if not self._trained:
            self.setup()

        home_id = self.repo.resolve_team_name(home_name)
        away_id = self.repo.resolve_team_name(away_name)

        unknown = [name for name, tid in ((home_name, home_id), (away_name, away_id)) if tid is None]
        if unknown:
            suggestions = {name: self.suggest(name) for name in unknown}
            return {
                "ok": False,
                "error": "unknown_team",
                "unknown": unknown,
                "suggestions": suggestions,
            }

        result = self.model.predict(home_id, away_id)
        strength = result["strength"]
        if strength["home"] > strength["away"]:
            stronger = home_name
        elif strength["away"] > strength["home"]:
            stronger = away_name
        else:
            stronger = None

        return {
            "ok": True,
            "home": home_name,
            "away": away_name,
            "predicted": result["predicted"],
            "probabilities": result["probabilities"],
            "strength": strength,
            "stronger_team": stronger,
            "score": result["score"],
        }
