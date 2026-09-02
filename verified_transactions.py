"""NFL-confirmed 2026 player trades, audited through September 1, 2026.

Source of truth: NFL.com league transaction logs for March, April, May, June,
July, and August 2026. This supplements nflverse when its transaction or roster
snapshots have not yet caught up.

This list intentionally includes depth-player trades, not only headline moves.
Draft-pick-only trades are handled by the draft dataset rather than this
player-movement supplement.
"""

VERIFIED_PLAYER_TRADES = [
    # March 2026
    {"season": 2026, "date": "2026-03-24", "player_name": "Andy Dalton", "position": "QB", "from_team": "CAR", "to_team": "PHI"},
    {"season": 2026, "date": "2026-03-23", "player_name": "Sydney Brown", "position": "S", "from_team": "PHI", "to_team": "ATL"},
    {"season": 2026, "date": "2026-03-19", "player_name": "Justin Fields", "position": "QB", "from_team": "NYJ", "to_team": "KC"},
    {"season": 2026, "date": "2026-03-18", "player_name": "Jaylen Waddle", "position": "WR", "from_team": "MIA", "to_team": "DEN"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Jermaine Johnson II", "position": "EDGE", "from_team": "NYJ", "to_team": "TEN"},
    {"season": 2026, "date": "2026-03-11", "player_name": "T'Vondre Sweat", "position": "DT", "from_team": "TEN", "to_team": "NYJ"},
    {"season": 2026, "date": "2026-03-11", "player_name": "DJ Moore", "position": "WR", "from_team": "CHI", "to_team": "BUF"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Garrett Bradbury", "position": "C", "from_team": "NE", "to_team": "CHI"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Colby Wooden", "position": "DT", "from_team": "GB", "to_team": "IND"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Zaire Franklin", "position": "LB", "from_team": "IND", "to_team": "GB"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Taron Johnson", "position": "CB", "from_team": "BUF", "to_team": "LV"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Rashan Gary", "position": "EDGE", "from_team": "GB", "to_team": "DAL"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Michael Pittman", "position": "WR", "from_team": "IND", "to_team": "PIT"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Geno Smith", "position": "QB", "from_team": "LV", "to_team": "NYJ"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Trent McDuffie", "position": "CB", "from_team": "KC", "to_team": "LA"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Kai Kroeger", "position": "P", "from_team": "NO", "to_team": "HOU"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Juice Scruggs", "position": "OL", "from_team": "HOU", "to_team": "DET"},
    {"season": 2026, "date": "2026-03-11", "player_name": "David Montgomery", "position": "RB", "from_team": "DET", "to_team": "HOU"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Minkah Fitzpatrick", "position": "S", "from_team": "MIA", "to_team": "NYJ"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Tytus Howard", "position": "OT", "from_team": "HOU", "to_team": "CLE"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Solomon Thomas", "position": "DT", "from_team": "DAL", "to_team": "TEN"},
    {"season": 2026, "date": "2026-03-11", "player_name": "Osa Odighizuwa", "position": "DT", "from_team": "DAL", "to_team": "SF"},

    # April 2026
    {"season": 2026, "date": "2026-04-25", "player_name": "Tyree Wilson", "position": "EDGE", "from_team": "LV", "to_team": "NO"},
    {"season": 2026, "date": "2026-04-24", "player_name": "Jonathan Greenard", "position": "EDGE", "from_team": "MIN", "to_team": "PHI"},
    {"season": 2026, "date": "2026-04-24", "player_name": "Dee Winters", "position": "LB", "from_team": "SF", "to_team": "DAL"},
    {"season": 2026, "date": "2026-04-20", "player_name": "Dexter Lawrence", "position": "DT", "from_team": "NYG", "to_team": "CIN"},
    {"season": 2026, "date": "2026-04-17", "player_name": "Maason Smith", "position": "DL", "from_team": "JAX", "to_team": "ATL"},
    {"season": 2026, "date": "2026-04-17", "player_name": "Ruke Orhorhoro", "position": "DL", "from_team": "ATL", "to_team": "JAX"},
    {"season": 2026, "date": "2026-04-13", "player_name": "Dontayvion Wicks", "position": "WR", "from_team": "GB", "to_team": "PHI"},
    {"season": 2026, "date": "2026-04-07", "player_name": "Marte Mapu", "position": "LB", "from_team": "NE", "to_team": "HOU"},

    # May 2026
    {"season": 2026, "date": "2026-05-27", "player_name": "Irvin Charles", "position": "WR", "from_team": "NYJ", "to_team": "SEA"},

    # June 2026
    {"season": 2026, "date": "2026-06-12", "player_name": "Wanya Morris", "position": "OT", "from_team": "KC", "to_team": "ATL"},
    {"season": 2026, "date": "2026-06-02", "player_name": "Myles Garrett", "position": "EDGE", "from_team": "CLE", "to_team": "LA"},
    {"season": 2026, "date": "2026-06-02", "player_name": "Jared Verse", "position": "EDGE", "from_team": "LA", "to_team": "CLE"},
    {"season": 2026, "date": "2026-06-02", "player_name": "A.J. Brown", "position": "WR", "from_team": "PHI", "to_team": "NE"},

    # July 2026: NFL transaction log lists no player trades.

    # August 2026
    {"season": 2026, "date": "2026-08-30", "player_name": "Broderick Jones", "position": "OT", "from_team": "PIT", "to_team": "DAL"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Diego Pounds", "position": "OT", "from_team": "BAL", "to_team": "KC"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Quinn Ewers", "position": "QB", "from_team": "MIA", "to_team": "JAX"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Sedrick Van Pran-Granger", "position": "C", "from_team": "BUF", "to_team": "IND"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Clark Phillips III", "position": "CB", "from_team": "ATL", "to_team": "CHI"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Gervon Dexter Sr.", "position": "DT", "from_team": "CHI", "to_team": "ATL"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Kyle McCord", "position": "QB", "from_team": "GB", "to_team": "MIA"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Hunter Long", "position": "TE", "from_team": "JAX", "to_team": "ARI"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Avery Smith", "position": "DB", "from_team": "LAC", "to_team": "SEA"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Kaleb Johnson", "position": "RB", "from_team": "PIT", "to_team": "GB"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Mark Redman", "position": "TE", "from_team": "LA", "to_team": "GB"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Jordan Meredith", "position": "OL", "from_team": "LV", "to_team": "NYJ"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Nathan Thomas", "position": "OT", "from_team": "DAL", "to_team": "HOU"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Jarrett Patterson", "position": "OL", "from_team": "HOU", "to_team": "SF"},
    {"season": 2026, "date": "2026-08-30", "player_name": "TeRah Edwards", "position": "DT", "from_team": "LAC", "to_team": "CAR"},
    {"season": 2026, "date": "2026-08-30", "player_name": "Walter Rouse", "position": "OT", "from_team": "MIN", "to_team": "NE"},
    {"season": 2026, "date": "2026-08-29", "player_name": "Basil Okoye", "position": "OT", "from_team": "BAL", "to_team": "NYG"},
    {"season": 2026, "date": "2026-08-29", "player_name": "Corey Kiner", "position": "RB", "from_team": "ARI", "to_team": "NE"},
    {"season": 2026, "date": "2026-08-27", "player_name": "Tutu Atwell", "position": "WR", "from_team": "MIA", "to_team": "LA"},
    {"season": 2026, "date": "2026-08-27", "player_name": "Jarquez Hunter", "position": "RB", "from_team": "LA", "to_team": "MIA"},
    {"season": 2026, "date": "2026-08-26", "player_name": "Joshua Ezeudu", "position": "OL", "from_team": "NYG", "to_team": "KC"},
    {"season": 2026, "date": "2026-08-24", "player_name": "Zamir White", "position": "RB", "from_team": "SF", "to_team": "NO"},
    {"season": 2026, "date": "2026-08-24", "player_name": "Deion Jones", "position": "LB", "from_team": "NO", "to_team": "SF"},
    {"season": 2026, "date": "2026-08-24", "player_name": "Kayshon Boutte", "position": "WR", "from_team": "NE", "to_team": "HOU"},
    {"season": 2026, "date": "2026-08-24", "player_name": "Jaylen Reed", "position": "DB", "from_team": "HOU", "to_team": "NE"},
    {"season": 2026, "date": "2026-08-19", "player_name": "Daniel Faalele", "position": "OT", "from_team": "NYG", "to_team": "JAX"},
    {"season": 2026, "date": "2026-08-10", "player_name": "Caedan Wallace", "position": "OT", "from_team": "NE", "to_team": "MIA"},
]
