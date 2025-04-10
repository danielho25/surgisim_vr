import csv
import random
import datetime
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

sample_data = """question,answer,score
What is the term for the process of making an incision in the skin during surgery?,Incision,0
"What does the prefix ""hypo-"" mean in medical terminology?",Below normal or deficient,0
What body system includes the heart and blood vessels?,Cardiovascular system,0
"What does the suffix ""-itis"" indicate in medical terminology?",Inflammation,0
What is the medical term for a blood clot?,Thrombus,0
What organ is responsible for producing insulin?,Pancreas,0
"What does the term ""bilateral"" mean in medical context?",Affecting both sides,0
What is the membrane that surrounds the lungs called?,Pleura,1
What is the medical term for excessive thirst?,Polydipsia,1
"What does the prefix ""tachy-"" refer to in medical terminology?",Rapid or fast,1
What is the medical term for the voice box?,Larynx,1
What is the medical term for the process of converting food into energy?,Metabolism,1
"What does the term ""idiopathic"" mean in medical diagnosis?",Of unknown cause,1
What is the medical term for the collarbone?,Clavicle,1
What is the name of the membrane that surrounds the heart?,Pericardium,2
"What does the term ""sequela"" refer to in medical context?",A condition that follows as a consequence of a disease,2
What is the medical term for the surgical creation of an opening between two hollow organs?,Anastomosis,2
"What does the term ""prodromal"" refer to in disease progression?",Early symptoms indicating onset of disease,2
What is the term for abnormal backward flow of blood through a valve?,Regurgitation,2
What is the medical term for the layer between the epidermis and subcutaneous tissue?,Dermis,2"""


class ml_quiz:
    def __init__(self, quiz_data, training_data):
        # questions organized by difficulty
        self.questions_by_difficulty = {
            0: [],
            1: [],
            2: []
        }

        # model state management
        self.ml_model = None  # store the machine learning model
        self.feature_scaler = StandardScaler()  # normalize data
        self.current_difficulty = 0  # current difficulty of the question
        self.score = 0  # users current score
        self.num_questions_asked = 0  # number of questions asked

        self.training_data_path = training_data

        # feature extraction
        self.session_details = []  # stores the details of the session history (question content, user answers, correct or not
        self.session_start_time = None  # tracks session start time
        self.session_end_time = None

        # feature extraction (tracks user performance)
        self.user_features = {
            'correct_ratio': 0.0,  # ratio of correct questions
            'avg_response_time': 0.0,  # response time of each question in seconds
            'consecutive_correct': 0,  # how many questions correct in a row
            'topic_performance': {},  # how well a user does in each topic
            'difficulty_performance': {0: 0.5, 1: 0.5, 2: 0.5}  #
        }

        # load the questions from the provided data
        self.load_questions(quiz_data)

        # initialize and train model if the is data
        if training_data:
            self.initialize_ml_model(training_data)

    # load the questions and parse data from csv
    def load_questions(self, data):
        # create a csv reader object
        csv_reader = csv.reader(StringIO(data))  # create a new csv reader object
        next(csv_reader)  # skip the headers in the csv

        question_id = 0  # counter to assign a number to each question
        for row in csv_reader:
            question, answer, difficulty = row
            difficulty = int(difficulty)  # cast the difficulty to an int

            # extract topic using the keywords
            topic = self.extract_topic(question)

            # store the questions with a unique id, text, correct answer, difficulty, topic area
            self.questions_by_difficulty[difficulty].append(
                {
                    'id': question_id,
                    'question': question,
                    'answer': answer,
                    'difficulty': difficulty,
                    'topic': topic,
                    'exposure_count': 0,  # how many times this question is shown
                    'correct_count': 0,  # how many times a question has been answered correctly
                    'avg_response_time': 0
                }
            )
            question_id += 1  # increment the question count by 1 (moves to the next question)

    def extract_topic(self, question_text):   # method to extract the topic based on the keywords in the text
        # dict of keywords
        keywords = {
            'anatomy': ['membrane', 'organ', 'tissue', 'bone', 'vessel', 'heart', 'lung', 'blood'],
            'terminology': ['term', 'prefix', 'suffix', 'mean', 'call', 'refer'],
            'physiology': ['process', 'function', 'responsible', 'energy', 'flow'],
            'pathology': ['disease', 'condition', 'abnormal', 'clot', 'inflammation']
        }

        # create a new dictionary called topic_scores using the keys, but all values are 0.
        # this dictionary tracks how many keywords appear in each topic
        topic_scores = {topic: 0 for topic in keywords}

        question_lower = question_text.lower()  # set the question to lowercase
        for topic, topic_keywords in keywords.items():  # iterate through each topic and the keywords in them
            for keyword in topic_keywords:  # go through each keyword in the current list of keywords
                if keyword in question_lower:  # if the keyword is in the question
                    topic_scores[topic] += 1  # increment the score fot that topic by 1

        # convert the dict to a list (.items() displays a list of the key values of a dictionary)
        # evaluate each tuple by looking at the score of the question and return the tuple with the highest score
        max_topic = max(topic_scores.items(), key=lambda x: x[1])  # use a lambda function to compare to the second element of each tuple
        if max_topic[1] > 0:  # return the name of the highest scoring topic (first element in tuple)
            return max_topic[0]
        else:
            "general"


    def initialize_ml_model(self, training_data_path):
        try:
            training_features, training_labels = self.load_training_data(training_data_path)

            # split the features for model performance, train the model on one set of data
            X_train, X_test, y_train, y_test = train_test_split(training_features, training_labels, test_size=0.2, random_state=42)

            # scale features
            self.feature_scaler.fit(X_train)
            X_train_scaled = self.feature_scaler.transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)

            # set up the random forest classifier ml model
            self.ml_model = RandomForestClassifier(
                n_estimators=100,  # use 100 decision trees
                max_depth=10,  # limit each tree to a depth of 10 nodes
                min_samples_split=5,  # require 5 samples to split a node
                random_state=42
            )
            self.ml_model.fit(X_train_scaled, y_train)  # train the model using the scaled x data and labels for that data

            # evaluate performance of the model
            y_pred = self.ml_model.predict(X_test_scaled)
            model_accuracy = accuracy_score(y_test, y_pred)
            print("Model Initialized!")
            print(f"model accuracy: {model_accuracy:.2f}")
            return True
        except Exception as e:
            print(f"Error Initialzing Model: {e}")
            print("Using rule based difficulty")
            return False


    def load_training_data(self, filepath):
        try:
            if not os.path.exists(filepath):  # check if the file exists
                print("Error, No path found")
                return self.generate_sample_data()

            features = []  # store the data
            labels = []

            with open(filepath, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    feature_vector = [
                        float(row['correct_ratio']),
                        float(row['avg_response_time']),
                        int(row['consecutive_correct']),
                        float(row['topic_expertise']),
                        int(row['difficulty'])
                    ]
                    features.append(feature_vector)

                    label = int(row['optimal_difficulty'])
                    labels.append(label)

                if not features: # if there are no training data csv
                    print("No valid data found!")
                    return self.generate_sample_data()
                print(f"training data loaded with {len(features)} samples")
                return np.array(features), np.array(labels)
        except Exception as e:
            print(f"error loading data: {e}")
            return self.generate_sample_data()

    def generate_sample_data(self):
        print("Generating simulated data...")

        features = np.random.rand(1000, 5)
        features[:, 4] = np.floor(features[:, 4] * 3)
        labels = np.zeros(1000)

        for i in range(1000):
            correct_ratio = features[i, 0]
            response_time = features[i, 1]
            difficulty = features[i, 4]

            if correct_ratio > 0.8 and response_time < 0.4:
               labels[i] = 0 if difficulty < 1.5 else 1
            elif correct_ratio < 0.4 or response_time > 0.8:
                labels[i] = 2 if difficulty > 0.5 else 1
            else:
                labels[i] = 1

        print(f"simulated training data generated: {len(features)}")
        return features, labels


    def extract_features(self, question, user_answer, response_time, is_correct):
        if is_correct:
            self.user_features['consecutive_correct'] += 1
        else:
            self.user_features['consecutive_correct'] += 0

        topic = question['topic']
        if topic not in self.user_features['topic_performance']:
            self.user_features['topic_performance'][topic] = 0.5

        alpha = 0.3
        self.user_features['topic_performance'][topic] = (
            alpha * (1 if is_correct else 0) +
            (1 - alpha) * self.user_features['topic_performance'][topic]
        )

        difficulty = question['difficulty']
        self.user_features['difficulty_performance'][difficulty] = (
            alpha * (1 if is_correct else 0) +
            (1 - alpha) * self.user_features['difficulty_performance'][difficulty]
        )

        total_questions = len(self.session_details)
        if total_questions > 0:
            correct_count = sum(1 for q in self.session_details if q['is_correct'])
            self.user_features['correct_ratio'] = correct_count / total_questions

            # update avg response time with moving average
            self.user_features['avg_response_time'] = (
                    alpha * response_time +
                    (1 - alpha) * self.user_features['avg_response_time']
            )

            feature_vector = [
                self.user_features['correct_ratio'],
                self.user_features['avg_response_time'],
                self.user_features['consecutive_correct'],
                self.user_features['topic_performance'].get(topic, 0.5),
                question['difficulty']
            ]
            return feature_vector

    def predict_optimal_difficulty(self, feature_vector):
        if self.ml_model is None:
            # Fall back to rule-based logic if no ML model is available
            return self.rule_based_difficulty_adjustment()

        try:
            if np.isnan(feature_vector).any():
                print("Nan values detected in feature vector! switching to rule based")
                return self.rule_based_difficulty_adjustment()


            feature_vector_2d = np.array(feature_vector).reshape(1,-1)
            # Scale the feature vector
            scaled_features = self.feature_scaler.transform([feature_vector])

            # Predict optimal difficulty category
            prediction = self.ml_model.predict(scaled_features)[0]

            # Adjust current difficulty based on prediction
            # 0 = too easy, 1 = appropriate, 2 = too hard
            if prediction == 0:  # Too easy
                return min(2, self.current_difficulty + 1)
            elif prediction == 2:  # Too hard
                return max(0, self.current_difficulty - 1)
            else:  # Appropriate
                return self.current_difficulty

        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return self.rule_based_difficulty_adjustment()

    def rule_based_difficulty_adjustment(self):
        correct_ratio = self.user_features['correct_ratio']

        if correct_ratio > 0.8:
            return min(2, self.current_difficulty + 1)
        elif correct_ratio < 0.4:
            return max(0, self.current_difficulty - 1)
        else:
            return self.current_difficulty

    def next_question(self):
        if not self.questions_by_difficulty[self.current_difficulty]:
            available_difficulties = []
            for difficulty_level, questions in self.questions_by_difficulty.items():
                if questions:  # if there is available questions in this level
                    available_difficulties.append(difficulty_level)

            if not available_difficulties:
                return None

            if self.ml_model:
                topic_strengths = self.user_features['topic_performance']

                weakest_topic = None
                lowest_score = float('inf')

                for topic, score in topic_strengths.items():
                    if score < lowest_score:
                        lowest_score = score
                        weakest_topic = topic

                questions_in_weakest_topic = []
                for difficulty in available_difficulties:
                    for q in self.questions_by_difficulty[difficulty]:
                        if weakest_topic and q['topic'] == weakest_topic:  # if the question is the weakest topic
                            questions_in_weakest_topic.append(q)

                if questions_in_weakest_topic:
                    selected_q = random.choice(questions_in_weakest_topic)
                    self.current_difficulty = selected_q['difficulty']
                    self.questions_by_difficulty[self.current_difficulty].remove(selected_q)
                    return selected_q

            self.current_difficulty = random.choice(available_difficulties)

        selected_question = random.choice(self.questions_by_difficulty[self.current_difficulty])  # get a random question from the available difficulties
        self.questions_by_difficulty[self.current_difficulty].remove(selected_question)
        return selected_question

    def check_answer(self, user_answer, correct_answer):
        return user_answer.lower().strip() == correct_answer.lower().strip()  # keywords

    def process_response(self, user_answer, correct_answer, response_time, question):
        is_correct = self.check_answer(user_answer, question['answer'])
        if is_correct:
            self.score += 1  # add 1 to the user score


        feature_vector = self.extract_features(question, user_answer, response_time, is_correct)
        # Record detailed interaction data
        interaction_data = {
            'question_id': question['id'],
            'question_text': question['question'],
            'difficulty_level': question['difficulty'],
            'topic': question['topic'],
            'user_answer': user_answer,
            'correct_answer': question['answer'],
            'is_correct': is_correct,
            'response_time': response_time,
            'cumulative_score': f"{self.score}/{self.num_questions_asked + 1}",
            'features': feature_vector
        }
        self.session_details.append(interaction_data)

        # Update question statistics
        question['exposure_count'] += 1
        if is_correct:
            question['correct_count'] += 1

        # Update average response time for this question
        alpha = 0.2  # Learning rate
        if question['avg_response_time'] == 0:
            question['avg_response_time'] = response_time
        else:
            question['avg_response_time'] = (
                    alpha * response_time +
                    (1 - alpha) * question['avg_response_time']
            )

        # Apply ML-based difficulty adjustment
        self.num_questions_asked += 1
        new_difficulty = self.predict_optimal_difficulty(feature_vector)

        # Log difficulty change for analysis
        difficulty_changed = new_difficulty != self.current_difficulty
        self.current_difficulty = new_difficulty

        return is_correct, difficulty_changed

    def save_results_to_csv(self, filename=None):
        if not self.session_details:
            print("No session data available to save!")
            return None

        # Generate default filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_quiz_results_{timestamp}.csv"

        try:
            with open(filename, 'w', newline='') as csvfile:
                # Define the CSV structure
                fieldnames = [
                    'question_number',
                    'question_id',
                    'question_text',
                    'topic',
                    'difficulty_level',
                    'user_answer',
                    'correct_answer',
                    'is_correct',
                    'response_time',
                    'cumulative_score',
                    'feature_correct_ratio',
                    'feature_avg_response_time',
                    'feature_consecutive_correct',
                    'feature_topic_performance'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Write each record to CSV
                question_number = 1
                for record in self.session_details:
                    row_data = {
                        'question_number': question_number,
                        'question_id': record['question_id'],
                        'question_text': record['question_text'],
                        'topic': record['topic'],
                        'difficulty_level': record['difficulty_level'],
                        'user_answer': record['user_answer'],
                        'correct_answer': record['correct_answer'],
                        'is_correct': record['is_correct'],
                        'response_time': f"{record['response_time']:.2f}",
                        'cumulative_score': record['cumulative_score'],
                        'feature_correct_ratio': f"{record['features'][0]:.3f}",
                        'feature_avg_response_time': f"{record['features'][1]:.3f}",
                        'feature_consecutive_correct': int(record['features'][2]),
                        'feature_topic_performance': f"{record['features'][3]:.3f}"
                    }
                    writer.writerow(row_data)
                    question_number += 1

                # Calculate and add summary statistics
                session_duration = 0
                if self.session_end_time and self.session_start_time:
                    session_duration = (self.session_end_time - self.session_start_time).total_seconds()

                final_score_percentage = 0
                if self.num_questions_asked > 0:
                    final_score_percentage = (self.score / self.num_questions_asked) * 100

                summary_row = {
                    'question_number': 'SUMMARY',
                    'question_text': f'Session Duration: {session_duration:.2f} seconds',
                    'difficulty_level': f'Final Difficulty: {self.current_difficulty}',
                    'is_correct': '',
                    'cumulative_score': f'Final Score: {self.score}/{self.num_questions_asked} ({final_score_percentage:.1f}%)'
                }
                writer.writerow(summary_row)

            print(f"Results successfully saved to {filename}")
            return filename

        except Exception as e:
            print(f"Error saving results: {e}")
            return None


    def update_model(self):
        if len(self.session_details) < 5:
            print("not enough data to satisfy model requirements!")
            return False
        try:
            appended_data = self.load_training_data(self.training_data_path)
            if not appended_data:
                print("error appending data")
                return False

            train_success = self.initialize_ml_model(self.training_data_path)
            if not train_success:
                print("failure to get training data!")

            save_success = self.save_model()
            if not save_success:
                print("Failed to save updated model")
            return False

            print("Model successfully updated with new session data")
            return True

        except Exception as e:
            print(f"Error updating ML model: {e}")
            return False

    def start_session(self, question_limit = 10):
        self.score = 0
        self.num_questions_asked = 0
        self.session_details = []
        self.session_start_time = datetime.datetime.now()

        print("Welcome to the quiz system! This quiz will learn from user responses and adapt difficulties based on user repsonses")

        while self.num_questions_asked < question_limit:
            question = self.next_question()
            if question is None:
                print("no questions available!")
                break

            print(f"\nQuestion {self.num_questions_asked + 1}: {question['question']}")
            print(f"Topic: {question['topic'].capitalize()} | Difficulty: {question['difficulty']}")

            question_start_time = datetime.datetime.now()
            user_answer = input("Your Answer: ")
            response_time = (datetime.datetime.now() - question_start_time)

            is_correct, difficulty_changed = self.process_response(user_answer, question['answer'], response_time.total_seconds(), question)

            if is_correct:
                print("✓ Correct!")
            else:
                print(f"✗ Incorrect. The correct answer is: {question['answer']}")

            # Provide adaptive feedback based on ML insights
            if difficulty_changed:
                if self.current_difficulty > question['difficulty']:
                    print("System has increased the difficulty level for you.")
                else:
                    print("System has adjusted to a more appropriate difficulty level.")

            print(f"Current score: {self.score}/{self.num_questions_asked} | Response time: {response_time.total_seconds():.2f}s")

        self.session_end_time = datetime.datetime.now()
        session_duration = (self.session_end_time - self.session_start_time).total_seconds()

        print("Session Complete!")
        if self.num_questions_asked > 0:
            percentage = (self.score / self.num_questions_asked) * 100
        else:
            percentage = 0

        print(f"Final score: {self.score}/{self.num_questions_asked} ({percentage:.1f}%)")
        print(f"Session Duration: {session_duration:.2f} seconds")
        print(f"Final Difficulty Level: {self.current_difficulty}")

        print("updating Model based on results!")
        self.update_model()

        save_results = input("\nWould you like to save your results to a CSV file? (y/n): ")
        if save_results.lower().startswith('y'):
            filename = 'ml_quiz_results.csv'
            self.save_results_to_csv(filename)


if __name__ == "__main__":
    # Initialize the ML quiz system with a proper file path
    training_data_file = "quiz_training_data.csv"

    # Check if the file exists, if not, use "simulated" which will generate sample data
    if not os.path.exists(training_data_file):
        print(f"Training data file {training_data_file} not found. Using simulated data instead.")
        training_data = "simulated"
    else:
        training_data = training_data_file

    quiz = ml_quiz(sample_data, training_data=training_data)
    quiz.start_session()

