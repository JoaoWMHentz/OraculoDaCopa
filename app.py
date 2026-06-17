import csv
import os
import sys

from service import Service

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _make_progress():
    # Barra de progresso animada (spinner + barra preenchida), atualizada na mesma linha.
    def render(done, total):
        frac = done / total
        width = 24
        filled = int(width * frac)
        bar = "█" * filled + "░" * (width - filled)
        spin = SPINNER[done % len(SPINNER)]
        end = "\n" if done >= total else ""
        sys.stdout.write(f"\r  {spin} [{bar}] {frac * 100:5.1f}% ({done}/{total}){end}")
        sys.stdout.flush()

    return render

LABEL_PT = {
    "home team win": "Vitória do mandante",
    "draw": "Empate",
    "away team win": "Vitória do visitante",
}


def _format_probabilities(res):
    # Cada linha mostra o resultado com o nome do time correspondente.
    rows = [
        (f"{LABEL_PT['home team win']} ({res['home']})", res["probabilities"]["home team win"]),
        (LABEL_PT["draw"], res["probabilities"]["draw"]),
        (f"{LABEL_PT['away team win']} ({res['away']})", res["probabilities"]["away team win"]),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [f"  {label + ':':<{width + 1}} {prob * 100:5.1f}%" for label, prob in rows]
    return "\n".join(lines)


def _predicted_label(res):
    # Resultado previsto: nome do time vencedor (ou "Empate").
    if res["predicted"] == "home team win":
        return res["home"]
    if res["predicted"] == "away team win":
        return res["away"]
    return LABEL_PT["draw"]


def _print_prediction(res):
    print(f"\n{res['home']} vs {res['away']}")
    print(f"Resultado previsto: {_predicted_label(res)}")
    score = res["score"]
    print(f"Placar estimado: {res['home']} {score['home_goals']} x {score['away_goals']} {res['away']}")
    print(_format_probabilities(res))
    if res["stronger_team"]:
        print(f"Historicamente mais forte: {res['stronger_team']}")
    else:
        print("Historicamente mais forte: equilibrado")
    print()


def _ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _confirm(prompt):
    answer = _ask(prompt).lower()
    return answer in ("y", "yes", "s", "sim")


def _resolve_name(service, name):
    # Retorna o nome válido; se errado, oferece a melhor sugestão (Y/N).
    # None significa "cancelar" (não encontrado ou usuário recusou).
    if service.is_known(name):
        return name
    similar = service.suggest(name)
    if similar and _confirm(f"Time '{name}' não encontrado. Você quis dizer: {similar[0]}? (Y/N) "):
        return similar[0]
    if not similar:
        sample = sorted(service.list_team_names())[:10]
        print("Exemplos de nomes válidos: " + ", ".join(sample) + " ...")
    return None


def run_cli(service):
    print("Oráculo da Copa — previsor de partidas da Copa do Mundo")
    print("Informe o time mandante e o visitante (pelo nome). Deixe em branco para sair.\n")

    while True:
        home = _ask("Time mandante: ")
        if not home:
            break
        home = _resolve_name(service, home)
        if home is None:
            print()
            continue

        away = _ask("Time visitante: ")
        if not away:
            break
        away = _resolve_name(service, away)
        if away is None:
            print()
            continue

        res = service.predict(home, away)
        _print_prediction(res)

    print("Até logo.")


def _real_result(home_goals, away_goals):
    if home_goals > away_goals:
        return "home team win"
    if away_goals > home_goals:
        return "away team win"
    return "draw"


def _load_test_matches(path):
    matches = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            matches.append({
                "date": row["date"],
                "home": row["home"],
                "away": row["away"],
                "home_goals": int(row["home_goals"]),
                "away_goals": int(row["away_goals"]),
                "result": _real_result(int(row["home_goals"]), int(row["away_goals"])),
            })
    return matches


def run_test(service, csv_path):
    matches = _load_test_matches(csv_path)
    print(f"\nTestando {len(matches)} jogos reais da Copa do Mundo 2026...\n")
    print(f"{'#':<3} {'Jogo':<42} {'Real':<22} {'Previsto':<22} {'Placar real':<13} {'Placar prev.':<13} {'V?':<4} {'P?'}")
    print("-" * 125)

    winner_hits = 0
    score_hits = 0
    skipped = []

    for i, m in enumerate(matches, 1):
        res = service.predict(m["home"], m["away"])
        if not res["ok"]:
            unknown = ", ".join(res["unknown"])
            game_label = f"{m['home']} vs {m['away']}"
            print(f"{i:<3} {game_label:<42} {'— sem histórico no banco (' + unknown + ')'}")
            skipped.append(m)
            continue

        real_result = m["result"]
        predicted = res["predicted"]
        pred_score = res["score"]

        hit_winner = real_result == predicted
        hit_score = (pred_score["home_goals"] == m["home_goals"] and
                     pred_score["away_goals"] == m["away_goals"])

        if hit_winner:
            winner_hits += 1
        if hit_score:
            score_hits += 1

        real_label = LABEL_PT[real_result]
        pred_label = LABEL_PT[predicted]
        real_score_str = f"{m['home_goals']} x {m['away_goals']}"
        pred_score_str = f"{pred_score['home_goals']} x {pred_score['away_goals']}"
        v_mark = "✓" if hit_winner else "✗"
        p_mark = "✓" if hit_score else "✗"

        game_label = f"{m['home']} vs {m['away']}"
        print(f"{i:<3} {game_label:<42} {real_label:<22} {pred_label:<22} {real_score_str:<13} {pred_score_str:<13} {v_mark:<4} {p_mark}")

    counted = len(matches) - len(skipped)
    print("-" * 125)
    print(f"\nResultados ({counted} jogos avaliados, {len(skipped)} pulados por time sem histórico):")
    print(f"  Acerto de vencedor: {winner_hits}/{counted} — {winner_hits / counted * 100:.1f}%")
    print(f"  Acerto de placar:   {score_hits}/{counted} — {score_hits / counted * 100:.1f}%")
    if skipped:
        skipped_names = ", ".join(f"{m['home']} vs {m['away']}" for m in skipped)
        print(f"\n  Jogos pulados: {skipped_names}")
    print()


def main():
    deep = "--deep-tune" in sys.argv
    tune = "--tune" in sys.argv
    test = "--test" in sys.argv
    service = Service()
    if deep:
        print("Treino profundo (busca extensa de hiperparâmetros, pode demorar)...")
        best = service.setup(deep=True, progress_callback=_make_progress())
        print(f"Melhor var_smoothing: {best['var_smoothing']:.2e}")
        print(f"Priors: {best['priors']}")
        print(f"Acurácia (validação cruzada 10-fold x5): {best['accuracy'] * 100:.1f}%")
    elif tune:
        print("Otimizando o modelo (busca de hiperparâmetros)...")
        best = service.setup(tune=True)
        print(f"Melhor var_smoothing: {best['var_smoothing']:.0e}")
        print(f"Acurácia (validação cruzada 5-fold): {best['accuracy'] * 100:.1f}%")
    else:
        print("Treinando o modelo com dados históricos da Copa do Mundo...")
        service.setup()

    if "--eval" in sys.argv:
        acc = service.evaluate()
        print(f"Acurácia (validação cruzada 5-fold): {acc * 100:.1f}%")

    if test:
        csv_path = next(
            (sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--test" and i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--")),
            os.path.join(os.path.dirname(__file__), "test_matches.csv"),
        )
        run_test(service, csv_path)
    else:
        run_cli(service)


if __name__ == "__main__":
    main()
