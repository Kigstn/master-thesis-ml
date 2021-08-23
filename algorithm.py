import imp
import os
import csv
import pickle
import random

from sklearn.ensemble import RandomForestClassifier


def save_data(use_case_id: int, use_case_step: int, user_emotion: str, user_emotion_reason: str, erg: str, evaluation: list) -> None:
    """
    Saves the evaluation to a file containing all previous evaluations

    A new dataset is created for every emotion / emotion reason combination
    A data point consist of all the ergs and if they were used (0-1) and the evaluation
    Machine Learning works with numbers, and this way we can represent which number was used when
    This way, we predict what erg works best for every distinct situation (for example use case 1, use case step 2, angry, retail is the reason), once we have enough data on each erg

    This also means, that the model needs a lot of runtime, to get starting data
    No data normalisation is needed, since all data points are 1-5
    """

    file_path = f"/data/{use_case_id}_{use_case_step}_{user_emotion}_{user_emotion_reason}.csv"

    # import all possible ergs for this emotion
    ergs = imp.load_source(f"{user_emotion}_{user_emotion_reason}", f"./ergs/{user_emotion}_{user_emotion_reason}.py").ergs

    # loop though the ergs and set them to 0 or 1, as explained before
    ergs_as_numbers = []
    for e in ergs:
        if e == erg:
            ergs_as_numbers.append(1)
        else:
            ergs_as_numbers.append(0)

    # insert the ergs into the row, since that's the target
    row = ergs_as_numbers + evaluation

    # set the mode to write if the file doesnt exist yet, else append
    if os.path.exists(file_path):
        mode = "a"
    else:
        mode = "w+"

    # open the file and input the data
    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def train_algorithm(use_case_id: int, use_case_step: int, user_emotion: str, user_emotion_reason: str) -> None:
    """
    To train the algorithm, first we check if there are at least 20 data points for every ERG
    This means there we need a lot of starting data to start this process
    It just doesn't make sense to have a ML algo work with less data, we would get very unrealistic results which are highly over-fitted

    If we have enough data, we split the data 50/50 into a training a testing set
    The training data will be used to train the model, and the testing data to test the model to prevent it from getting over-fitted to our training data
    """

    file_path = f"/data/{use_case_id}_{use_case_step}_{user_emotion}_{user_emotion_reason}.csv"

    # import all possible ergs for this emotion
    ergs = imp.load_source(f"{user_emotion}_{user_emotion_reason}", f"./ergs/{user_emotion}_{user_emotion_reason}.py").ergs

    # look if min 20 data points exist for ever erg in the data
    if not os.path.exists(file_path):
        return
    else:
        with open(file_path, "r", newline='') as file:
            reader = csv.reader(file)

            # loop through each erg
            for erg in ergs:
                count = 0

                # get the index of the erg in the dataset, we need to know where we have to look
                index = ergs.index(erg)

                # loop through each datapoint
                for row in reader:
                    # test if the erg is the one we are looking for. reminder: 1 means erg was used, 0 means it wasn't
                    if row[index] == 1:
                        count += 1

                # test if we have enough
                if count < 20:
                    return

    # we validated that we have enough data
    # so lets train it
    # using a simple random forest classifier for this, can adapt to any wanted model
    # since this is just a prototype
    clf = RandomForestClassifier(random_state=0)

    # get all rows as a 2d array split into x and y
    # x is the train data and y the wanted output
    x = []
    y = []
    with open(file_path, "r", newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # now we split the row in half, exactly at the point were the ergs end and the evaluation data starts
            # don't forget arrays start at 0, so we need to subtract 1
            x.append(row[len(ergs) - 1:])
            y.append(row[:len(ergs) - 1])

    # train the model
    clf.fit(x, y)

    # save the model in a pickle, since we want to use it in a different function call
    with open(f"./models/{use_case_id}_{use_case_step}_{user_emotion}_{user_emotion_reason}.pickle", "wb+") as file:
        pickle.dump(clf, file)


def predict_ergs(use_case_id: int, use_case_step: int, user_emotion: str, user_emotion_reason: str) -> list[str]:
    """
    Use the existing model to predict what 5 ergs to give to the user
    The ergs with the best evaluation for the specific situation are used

    If no model exists, get 5 randomly to generate more training data
    """

    ergs = imp.load_source(f"{user_emotion}_{user_emotion_reason}", f"./ergs/{user_emotion}_{user_emotion_reason}.py").ergs

    # look if the model exists
    model_path = f"./models/{use_case_id}_{use_case_step}_{user_emotion}_{user_emotion_reason}.pickle"
    if os.path.exists(model_path):
        # load the model from the pickle
        with open(model_path, "wb+") as file:
            clf = pickle.load(file)

        # predict the ergs
        # for that we loop through all possible ergs and get their score
        # at the end, the best five get returned

        results = []
        for erg in ergs:
            # construct the input list which has is 0s and one 1 where the erg is normally
            input_list = [0 for _ in ergs]
            input_list[ergs.index(erg)] = 1

            # predict it with that
            results.append(clf.predict(input_list))

        # sort that
        sorted_results = sorted(results, reverse=True)

        # get the best 5
        best_five = sorted_results[:5]

        # loop through the ergs again to get the name of the erg
        best_five_name = [ergs[index] for index in best_five]

        return best_five_name

    else:
        # if the model doesnt exist, return five random one
        five_random = [random.choice(ergs) for _ in range(5)]

        return five_random
