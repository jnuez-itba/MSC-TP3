from source.dam.study import SWEEPS
from source.dam.study import make_variant
from source.dam.study import evaluate
from source.dam.study import run_study
from source.dam.study import best_at_area


def test_variant_areas():
    # base B: (B + 3)/2 x 30; talón a: (18 + a + 3)/2 x 30; cresta c: (18 + c)/2 x 30;
    # talud con misma base: siempre 315
    assert abs(make_variant("base", 22.0).area() - 375.0) < 1.0e-9
    assert abs(make_variant("talon", 4.0).area() - 375.0) < 1.0e-9
    assert abs(make_variant("cresta", 7.0).area() - 375.0) < 1.0e-9

    for a in SWEEPS["talud"]:
        assert abs(make_variant("talud", a).area() - 315.0) < 1.0e-9


def test_toe_stays_for_heel_extension():
    for a in SWEEPS["talon"]:
        geometry = make_variant("talon", a)
        assert abs(geometry.toe()[0] - 18.0) < 1.0e-12
        assert abs(geometry.heel()[0] + a) < 1.0e-12


def test_safety_factor_trends():
    results = run_study(fem=False)

    # crece al alargar la base, al extender el talón y al ensanchar la cresta
    for family in ["base", "talon", "cresta"]:
        factors = [r["safety_factor"] for v, r in results[family]]
        assert all(factors[k + 1] > factors[k] for k in range(len(factors) - 1))

    # inclinar el paramento sin agregar hormigón lo baja
    factors = [r["safety_factor"] for v, r in results["talud"]]
    assert all(factors[k + 1] < factors[k] for k in range(len(factors) - 1))


def test_same_area_ranking():
    results = run_study(fem=False)
    chosen = best_at_area(results, 375.0)

    assert [family for family, value, r in chosen] == ["base", "talon", "cresta"]

    # el talón extendido es el que menos fricción necesita
    friction = {family: r["required_friction"] for family, value, r in chosen}
    assert friction["talon"] < friction["base"]
    assert friction["talon"] < friction["cresta"]


def test_fem_matches_hand_for_variants():
    for family, value in [("base", 22.0), ("talon", 4.0), ("cresta", 7.0), ("talud", 4.0)]:
        result = evaluate(make_variant(family, value), fem=True, element_size=3.0)

        assert result["fem_moment_error"] < 1.0e-8
        assert result["ux_crest"] > 0.0
