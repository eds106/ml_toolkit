import pytest
from src.ml_helpers import classification_metrics, generate_linear, generate_quadratic, generate_polynomial, generate_classes

@pytest.mark.parametrize("non_implemented_method",
[(classification_metrics),
(generate_linear),
(generate_quadratic),
(generate_polynomial),
(generate_classes)])
def test_non_implemented(non_implemented_method):
    with pytest.raises(NotImplementedError):
        non_implemented_method()
