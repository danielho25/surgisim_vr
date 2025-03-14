import numpy as np
from sklearn.ensemble import \
    RandomForestClassifier  # multiple decision trees trained on different parts of training data
from sklearn.preprocessing import OneHotEncoder  # convert categorical to numerical data for preprocessing


class question_model:
    def __init__(self):
        # initialize random forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.encoder = OneHotEncoder(sparse_output=False)
        self.is_trained = False

        # Define sample data (Change to actual used questions later)
        # stored format, current question rank (0 = easy, 2 = hard, is answer correct (0 = no, 1 = yes), previous q rank
        question_data = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [2, 1, 1],
            [1, 0, 2],
        ])

        next_difficulty = np.array([0, 0, 1, 0, 2, 0])
        self.train_model(question_data, next_difficulty)

    def train_model(self, data, next_d):
        self.model.fit(data, next_d)
        self.is_trained = True

    def next_question_difficulty(self, current_difficulty, correct_answer, previous_difficulty):
        # Check if the model is trained
        if not self.is_trained:
            # If not trained, use simple rule-based approach
            return min(current_difficulty + 1, 2)  # return the min of current level +1, with max of 2

        # If the model is trained, use it to predict the next difficulty
        input_data = np.array([[current_difficulty, correct_answer, previous_difficulty]])
        next_difficulty = self.model.predict(input_data)[0]
        return next_difficulty

    def update_data(self, new_data, new_difficulty):
        if self.is_trained:
            self.model.fit(new_data, new_difficulty)
        else:
            self.train_model(new_data, new_difficulty)

    def update_model(self, update_data, update_next_difficulty):
        self.update_data(np.array(update_data), np.array(update_next_difficulty))


def main():
    model = question_model()
    question_bank = {
        0: [
            "What is the name of the process of making an incision in the skin during surgery?",
            "Which instrument is commonly used to make surgical incisions?",
            "What is the medical term for stitching a wound closed?"
        ],
        1: [
            "What is the purpose of hemostats in surgery?",
            "Which type of anesthesia allows a patient to remain awake but pain-free during surgery?",
            "What is the difference between a laparoscopic and an open surgical procedure?"
        ],
        2: [
            "What are the main differences between absorbable and non-absorbable sutures, and when is each used?",
            "In what situations would a surgeon choose to perform a fasciotomy?",
            "What are the key steps in preventing surgical site infections (SSIs) in the operating room?"
        ]
    }

    # store the data in a session
    update_data = []
    update_next_difficulty = []

    current_diff = 1
    previous_diff = 0

    for i in range(3):
        q = np.random.choice(question_bank[current_diff])
        print(f"{q}  \nCurrent Difficulty: {current_diff}")
        # simulate an answer using 0 for incorrect and 1 for correct at random
        correct = np.random.choice([0, 1])
        if correct:
            print(f"you answered correctly!")
        else:
            print("sorry that is incorrect")

        # record the results of the session answers
        update_data.append([current_diff, correct, previous_diff])

        # predict the next questions difficulty
        next_diff = model.next_question_difficulty(current_diff, correct, previous_diff)
        print(f"Model recommends: {next_diff}") # debugging to see what the model will predict

        update_next_difficulty.append(next_diff)
        previous_diff = current_diff
        current_diff = next_diff

    model.update_data(update_data, update_next_difficulty)
    print("model updated!")

if __name__ == "__main__":
    main()