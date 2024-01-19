import numpy as np
import os
import uvicorn
import torch


# /src/nlp_engineer_assignment
from nlp_engineer_assignment import count_letters, print_line, read_inputs, \
    score, train_classifier, test_classifier


def train_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    ###
    # Setup
    ###

    # Constructs the vocabulary as described in the assignment
    vocabs = [chr(ord('a') + i) for i in range(0, 26)] + [' ']

    ###
    # Train
    ###

    train_inputs = read_inputs(
        os.path.join(cur_dir, "data", "train.txt")
    )
    
    
    model = train_classifier(train_inputs, vocabs, batch_size=32, EPOCHS=10)
    # model = train_classifier(train_inputs, vocabs, batch_size=32, EPOCHS=1)
    torch.save(model.state_dict(), "trained_model")

    ###
    # Test
    ###

    test_inputs = read_inputs(
        os.path.join(cur_dir, "data", "test.txt")
    )

    # TODO: Extract predictions from the model and save it to a
    # variable called `predictions`. Observe the shape of the
    # example random predictions.
    # golds = np.stack([count_letters(text) for text in test_inputs])
    # predictions = np.random.randint(0, 3, size=golds.shape)

    golds = np.stack([count_letters(text) for text in test_inputs])
    predictions = test_classifier(model, test_inputs=test_inputs, vocabulary=vocabs)



    # Print the first five inputs, golds, and predictions for analysis
    for i in range(5):
        print(f"Input {i+1}: {test_inputs[i]}")
        print(
            f"Gold {i+1}: {count_letters(test_inputs[i]).tolist()}"
        )
        print(f"Pred {i+1}: {predictions[i].tolist()}")
        print_line()

    print(f"Test Accuracy: {100.0 * score(golds, predictions):.2f}%")
    print_line()


if __name__ == "__main__":
    train_model()
    uvicorn.run(
        "nlp_engineer_assignment.api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1
    )
