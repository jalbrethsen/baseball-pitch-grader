bb_types = {"line_drive": 0, "ground_ball": 1, "fly_ball": 2, "popup": 3}
pitch_types = {
    "CH": 0,  # changeup
    "CU": 1,  # curveball
    "FC": 2,  # cutter
    "EP": 3,  # ephus
    "FO": 4,  # forkball
    "FF": 5,  # four seam fastball
    "KN": 6,  # knuckleball
    "KC": 7,  # knuckle-curve
    "SC": 8,  # screwball
    "SI": 9,  # sinker
    "SL": 10,  # slider
    "SV": 11,  # slurve
    "FS": 12,  # splitter
    "ST": 13,  # sweeper,
    "IN": 14,  # intentional ball
    "PO": 15,  # pitchout
}
events = {
    "catcher_interf": 0,
    "double": 1,
    "field_error": 2,
    "field_out": 3,
    "fielders_choice": 4,
    "fielders_choice_out": 5,
    "force_out": 6,
    "grounded_into_double_play": 7,
    "hit_by_pitch": 8,
    "home_run": 9,
    "sac_fly": 10,
    "single": 11,
    "strikeout": 12,
    "triple": 13,
    "walk": 14,
    "double_play": 15,
    "sac_fly_double_play": 16,
    "truncated_pa": 17,
    "sac_bunt": 18,
    "intent_walk": 19,
    "sac_fly_double_play": 20,
    "strikeout_double_play": 21,
}
outcomes = {
    "line_drive": 0,
    "ground_ball": 1,
    "fly_ball": 2,
    "popup": 3,
    "strikeout": 4,
    "walk": 5,
}
outcomes_v2 = [
    "walk",
    "single",
    "double",
    "triple",
    "home_run",
    "hit_by_pitch",
    "intent_walk",
]
righty_lefty = {"R": 1, "L": 0}
pitch_outcomes = {
    "S": 0,
    "B": 1,
    "X": 2,
}
