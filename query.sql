SELECT
    player_id, -- for drafting simulation only
    season, -- for drafting simulation only
    round, -- for drafting simulation only
    club, -- for drafting simulation only
    position, -- for drafting simulation only
    price_cartola_express AS price, -- for drafting simulation only
    position_id,
    spi_club,
    spi_opponent,
    prob_club,
    prob_opponent,
    prob_tie,
    importance_club,
    importance_opponent,
    proj_score_club,
    proj_score_opponent,
    total_points_club_last_19,
    offensive_points_club_last_19,
    defensive_points_club_last_19,
    total_allowed_points_opponent_last_19,
    offensive_allowed_points_opponent_last_19,
    defensive_allowed_points_opponent_last_19,
    total_points_club_last_5,
    offensive_points_club_last_5,
    defensive_points_club_last_5,
    total_allowed_points_opponent_last_5,
    offensive_allowed_points_opponent_last_5,
    defensive_allowed_points_opponent_last_5,
    avg_odds_club,
    avg_odds_opponent,
    avg_odds_draw,
    total_points_last_19,
    offensive_points_last_19,
    defensive_points_last_19,
    total_points_last_5,
    offensive_points_last_5,
    defensive_points_last_5,
    total_points -- target
FROM
    palpiteiro.fct_player
WHERE
    played IS TRUE