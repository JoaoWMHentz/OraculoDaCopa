import os
import sqlite3

DEFAULT_DB_PATH = "/home/joao/Documents/Repositories/Catolica/IA/worldcup/data-sqlite/worldcup.db"
WORLDCUP_DB = os.environ.get("WORLDCUP_DB", DEFAULT_DB_PATH)

# Janela histórica: ignora Copas anteriores a este ano (eras muito antigas distorcem
# a força atual dos times). 0 = sem limite. Pode ser sobrescrito por WORLDCUP_MIN_YEAR.
DEFAULT_MIN_YEAR = int(os.environ.get("WORLDCUP_MIN_YEAR", "2010"))


class Repository:
    """Only place in the project that touches the database."""

    def __init__(self, db_path=WORLDCUP_DB, min_year=DEFAULT_MIN_YEAR):
        self.db_path = db_path
        self.min_year = min_year
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row

    def _year_filter(self, alias):
        # Cláusula opcional que restringe a janela histórica via join com tournaments.
        if not self.min_year:
            return "", ()
        return (
            f"WHERE {alias}.tournament_id IN "
            "(SELECT tournament_id FROM tournaments WHERE year >= ?)",
            (self.min_year,),
        )

    def get_matches(self):
        where, params = self._year_filter("m")
        rows = self._conn.execute(
            f"""
            SELECT m.match_id, m.home_team_id, m.away_team_id,
                   m.home_team_score, m.away_team_score, m.result
            FROM matches m
            {where}
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_team_appearances(self):
        year_clause = "AND t.year >= ?" if self.min_year else ""
        params = (self.min_year,) if self.min_year else ()
        rows = self._conn.execute(
            f"""
            SELECT ta.match_id, ta.team_id, ta.opponent_id, ta.home_team,
                   ta.goals_for, ta.goals_against, ta.win, ta.lose, ta.draw,
                   t.year
            FROM team_appearances ta
            JOIN tournaments t ON ta.tournament_id = t.tournament_id
            WHERE 1=1 {year_clause}
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_teams(self):
        rows = self._conn.execute(
            "SELECT team_id, team_name, team_code FROM teams"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_team_names(self):
        rows = self._conn.execute(
            "SELECT team_name FROM teams ORDER BY team_name"
        ).fetchall()
        return [r["team_name"] for r in rows]

    def resolve_team_name(self, name):
        row = self._conn.execute(
            "SELECT team_id FROM teams WHERE LOWER(team_name) = LOWER(?)",
            (name.strip(),),
        ).fetchone()
        return row["team_id"] if row else None

    def close(self):
        self._conn.close()
