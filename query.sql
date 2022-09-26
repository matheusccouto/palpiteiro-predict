SELECT
    season, -- for drafting simulation only
    round, -- for drafting simulation only
    club, -- for drafting simulation only
    price_cartola_express AS price, -- for drafting simulation only
    position,
    total_points -- target
FROM
    palpiteiro.fct_player
WHERE
    played IS TRUE
ORDER BY
    timestamp