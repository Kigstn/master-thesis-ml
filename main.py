from algorithm import predict_ergs, save_data, train_algorithm


def get_use_case_response(use_case_id: int, use_case_step: int, user_emotion: str, user_emotion_reason: str) -> list[str]:
    """
    The main function has the same name and signature as in the master thesis (https://github.com/Kigstn/master-thesis/blob/main/app/use_case_response.py)
    This function also needs the users emotional state
    So they can be exchanged at will

    Returns five erg, which are evaluated by a ml algorithm (once enough data is present)
    """

    # make sure the emotions are valid
    assert user_emotion in ["happy", "angry", "anxious", "embarrassed", "relaxed", "sad"]
    assert user_emotion_reason in ["retail", "not retail"]

    # use the ml model to get the data
    return predict_ergs(
        use_case_id=use_case_id,
        use_case_step=use_case_step,
        user_emotion=user_emotion,
        user_emotion_reason=user_emotion_reason,
    )


def update_ml_model(use_case_id: int, use_case_step: int, user_emotion: str, user_emotion_reason: str, erg: str, evaluation: list) -> None:
    """
    This functions gets called after a use case response has been evaluated by the user. For example, by the first survey used in the master thesis
    The results from that evaluation get returned to this functions, to then update the model

    :evaluation is a list of results so, for example, [1, 5, 2, 3, 3, 4] and will always have the same structure.
    So question one which has the value 1 can be "Der Wert, den ich durch den Service vom Anbieter erhalte, ist die investierte Zeit und Mühe wert." (from the first survey of the master thesis) and so on. The value 1 means that the user rated the question with "Stimme überhaupt nicht zu"
    """

    # make sure the args are valid
    assert user_emotion in ["happy", "angry", "anxious", "embarrassed", "relaxed", "sad"]
    assert user_emotion_reason in ["retail", "not retail"]

    # save the data
    save_data(
        use_case_id=use_case_id,
        use_case_step=use_case_step,
        user_emotion=user_emotion,
        user_emotion_reason=user_emotion_reason,
        erg=erg,
        evaluation=evaluation
    )

    # train the algorithm
    train_algorithm(
        use_case_id=use_case_id,
        use_case_step=use_case_step,
        user_emotion=user_emotion,
        user_emotion_reason=user_emotion_reason,
    )

