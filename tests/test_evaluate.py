import numpy as np
import pytest

from dlquanttime.evaluate import evaluate_predictions


def test_evaluate_predictions_structure_and_values():
    metabolites = ("A", "B")
    y_true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    y_pred = y_true.copy()
    results = evaluate_predictions(y_true, y_pred, metabolites)

    assert set(results.keys()) == set(metabolites)
    for name in metabolites:
        assert set(results[name].keys()) == {"pearson_r", "icc"}
        assert results[name]["pearson_r"] == pytest.approx(1.0)
        assert results[name]["icc"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_predictions_maps_columns_independently():
    metabolites = ("A", "B")
    y_true = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    # Column A is perfectly predicted, column B is perfectly anti-correlated.
    y_pred = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    results = evaluate_predictions(y_true, y_pred, metabolites)

    assert results["A"]["pearson_r"] == pytest.approx(1.0)
    assert results["B"]["pearson_r"] == pytest.approx(-1.0)
