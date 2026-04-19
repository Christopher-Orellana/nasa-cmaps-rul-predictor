import math
import pytest

from src.inference.predict import predict_rul

def test_predict_rul_valid_input_returns_expected_schema():
    """This function validates and tests that the features are the expected datatype and order"""
    input_data = {
        'unit_number': 1,
        'sensor_4': 1410.5,
        'sensor_11': 47.6,
        'sensor_15': 8.45
    }

    output = predict_rul(input_data)

    # make sure output is a dictionary
    assert isinstance(output, dict)

    required_top_level_keys = {
        "unit_number",
        "timestamp",
        "artifact_version",
        "rul_pred",
        "rul_lower",
        "risk_band",
        "recommended_action",
        "flags"
    }

    # make sure output contains all the top level keys
    assert required_top_level_keys.issubset(output.keys())

    assert output['unit_number'] == input_data['unit_number']
    assert output["artifact_version"] == "baseline_v1"

    assert isinstance(output['rul_pred'], float)
    assert isinstance(output['rul_lower'], float)
    assert math.isfinite(output['rul_pred'])
    assert math.isfinite(output['rul_lower'])

    assert output['risk_band'] in {'GREEN', 'AMBER', 'RED', 'CRITICAL'}
    assert output['recommended_action'] in {
        'CONTINUE',
        'INSPECT',
        'SCHEDULE_MAINTENANCE',
        'REMOVE_FROM_SERVICE'
    }

    assert isinstance(output['flags'], dict)

    required_flag_keys = {
        'schema_error',
        'extrapolation_risk',
        'input_anomaly'
    }
    assert required_flag_keys.issubset(output['flags'].keys())

    assert isinstance(output['flags']['schema_error'], bool)
    assert isinstance(output['flags']['extrapolation_risk'], bool)
    assert isinstance(output['flags']['input_anomaly'], bool)


def test_predict_rul_missing_unit_number_raises_value_error():
    input_data  = {
        'sensor_4': 1410.5,
        'sensor_11': 47.6,
        'sensor_15': 8.45
    }

    with pytest.raises(ValueError, match="Missing required metadata"):
        predict_rul(input_data)


def test_predict_rul_extra_field_raises_value_error():
    input_data = {
        'unit_number': 1,
        'sensor_4': 1410.5,
        'sensor_11': 47.6,
        'sensor_15': 8.45,
        'sensor_99': 123.0
    }

    with pytest.raises(ValueError, match="Unexpected extra fields"):
        predict_rul(input_data)


def test_predict_rul_invalid_unit_number_type_raises_value_error():
    input_data = {
        'unit_number': '1',
        'sensor_4': 1410.5,
        'sensor_11': 47.6,
        'sensor_15': 8.45,
    }

    with pytest.raises(ValueError, match='unit_number must be an integer'):
        predict_rul(input_data)


def test_predict_rul_missing_required_feature_raises_value_error():
    input_data = {
        'unit_number': 1,
        'sensor_4': 1410.5,
        'sensor_11': 47.6
    }

    with pytest.raises(ValueError, match='Missing required features'):
        predict_rul(input_data)


def test_predict_rul_non_finite_sensor_value_raises_value_error():
    input_data = {
        'unit_number': 1,
        'sensor_4': float("nan"),
        'sensor_11': 47.6,
        'sensor_15': 8.45
    }

    with pytest.raises(ValueError, match='Input contains NaN or non-finite values'):
        predict_rul(input_data)