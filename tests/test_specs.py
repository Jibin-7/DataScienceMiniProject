from medisense.specs import SPECS, get_spec


def test_four_disease_scope_is_preserved():
    assert set(SPECS) == {"diabetes", "heart", "parkinsons", "breast_cancer"}


def test_every_form_feature_has_a_sensible_range():
    for spec in SPECS.values():
        assert spec.features
        assert all(feature.minimum < feature.maximum for feature in spec.features)
        assert get_spec(spec.key) == spec
