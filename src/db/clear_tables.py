from sqlalchemy import text

from src.db.session import get_engine


def clear_predictions_and_inputs() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        before_pred = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar() or 0
        before_inputs = conn.execute(text("SELECT COUNT(*) FROM inference_inputs")).scalar() or 0

        conn.execute(text("DELETE FROM predictions"))
        conn.execute(text("DELETE FROM inference_inputs"))

        after_pred = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar() or 0
        after_inputs = conn.execute(text("SELECT COUNT(*) FROM inference_inputs")).scalar() or 0

        print(f"predictions: {before_pred} -> {after_pred}")
        print(f"inference_inputs: {before_inputs} -> {after_inputs}")


if __name__ == "__main__":
    clear_predictions_and_inputs()


