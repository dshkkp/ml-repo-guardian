from model_io import load_model, save_model


def test_issue_04_save_model(tmp_path):
    """Сохранение модели должно создавать отсутствующие родительские папки."""

    model = {"version": 1}
    model_path = tmp_path / "nested" / "models" / "model.joblib"

    assert not model_path.parent.exists()

    save_model(model, model_path)

    assert model_path.is_file()
    assert load_model(model_path) == model
