import csv
import random
import datetime
from io import StringIO

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


class quiz_model:
    def __init__(self, data):
        # store the questions by difficulty
        self.question_difficulty = {
            0: [],
            1: [],
            2: []
        }

        # set initial difficulty to 0
        self.current_difficulty = 0
        self.score = 0
        self.questions_asked = 0

        # Add session tracking for detailed reporting
        self.session_details = []
        self.session_start_time = None
        self.session_end_time = None

        # read csv data
        csv_reader = csv.reader(StringIO(data))
        next(csv_reader)  # skip the header row

        # read the csv files
        for row in csv_reader:
            question, answer, difficulty = row
            # convert the difficulty into an int
            difficulty = int(difficulty)
            # store each question according to difficulty
            self.question_difficulty[difficulty].append({
                'question': question,
                'answer': answer,
                'difficulty': difficulty
            })

    # selects a new question based on difficulty
    def next_q(self):
        # if there are no questions at current difficulty level
        if not self.question_difficulty[self.current_difficulty]:
            available_question_difficulty = []  # list of available questions left based on difficulty

            for difficulty, questions in self.question_difficulty.items():
                if questions:  # if the list is not empty add it to the list of available questions
                    available_question_difficulty.append(difficulty)

            # Check if there are any questions available at all
            if not available_question_difficulty:
                return None  # Return None if no more questions are available

            # Choose a random difficulty level from available ones
            self.current_difficulty = random.choice(available_question_difficulty)

        # Pick a random question from current difficulty
        new_random_question = random.choice(self.question_difficulty[self.current_difficulty])

        # Remove the question from the list
        self.question_difficulty[self.current_difficulty].remove(new_random_question)
        return new_random_question

    def check_answer(self, user_answer, correct_answer):
        return user_answer.lower().strip() == correct_answer.lower().strip()

    def adjust_question_difficulty(self, is_correct):
        max_difficulty = max(self.question_difficulty.keys())
        min_difficulty = min(self.question_difficulty.keys())

        if is_correct:  # if the user is correct
            if self.current_difficulty < max_difficulty:  # if not at most difficult question
                self.current_difficulty += 1  # increase the difficulty
        else:  # if incorrect
            if self.current_difficulty > min_difficulty:
                self.current_difficulty -= 1  # decrease difficulty if not at lowest level

    def save_results_to_csv(self, filename=None):
        """
        Save the session results to a CSV file with comprehensive performance data.

        Args:
            filename: Optional filename for the CSV. If None, generates a timestamped filename.

        Returns:
            str: Path to the saved CSV file
        """
        if not self.session_details:
            print("No session data available to save!")
            return None

        # Generate default filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quiz_results_{timestamp}.csv"

        try:
            with open(filename, 'w', newline='') as csvfile:
                # Define CSV structure with comprehensive metrics
                fieldnames = [
                    'question_number',
                    'question_text',
                    'difficulty_level',
                    'user_answer',
                    'correct_answer',
                    'is_correct',
                    'cumulative_score'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Write each question's data
                for record in self.session_details:
                    writer.writerow(record)

                # Add summary metrics at the end
                writer.writerow({
                    'question_number': 'SUMMARY',
                    'question_text': f'Session Duration: {(self.session_end_time - self.session_start_time).total_seconds():.2f} seconds',
                    'difficulty_level': f'Final Difficulty: {self.current_difficulty}',
                    'user_answer': '',
                    'correct_answer': '',
                    'is_correct': '',
                    'cumulative_score': f'Final Score: {self.score}/{self.questions_asked} ({(self.score / self.questions_asked * 100) if self.questions_asked > 0 else 0:.1f}%)'
                })

            print(f"Results successfully saved to {filename}")
            return filename

        except Exception as e:
            print(f"Error saving results: {e}")
            return None

    def start_session(self, question_limit=5):
        self.score = 0
        self.questions_asked = 0
        self.session_details = []  # Reset session data
        self.session_start_time = datetime.datetime.now()  # Record start time

        print("Welcome to the Adaptive Surgery Quiz!")
        print("Answer the questions to test your medical knowledge.")
        print("Questions will adapt based on your performance.\n")

        while self.questions_asked < question_limit:
            new_random_question = self.next_q()

            # Check if we have a valid question
            if new_random_question is None:
                print("No more questions available!")
                break

            print(f"Question Difficulty: {new_random_question['difficulty']}")
            print(f"Question {self.questions_asked + 1}: {new_random_question['question']}")

            user_answer = input("Please enter your answer: ")
            is_correct = self.check_answer(user_answer, new_random_question['answer'])

            if is_correct:
                print("Correct!")
                self.score += 1
            else:
                print(f"Incorrect! Correct Answer: {new_random_question['answer']}")

            # Track detailed question data for reporting
            self.session_details.append({
                'question_number': self.questions_asked + 1,
                'question_text': new_random_question['question'],
                'difficulty_level': new_random_question['difficulty'],
                'user_answer': user_answer,
                'correct_answer': new_random_question['answer'],
                'is_correct': is_correct,
                'cumulative_score': f"{self.score}/{self.questions_asked + 1}"
            })

            self.questions_asked += 1
            self.adjust_question_difficulty(is_correct)
            print(f"Current score: {self.score}/{self.questions_asked}\n")

        self.session_end_time = datetime.datetime.now()  # Record end time

        print(f"\nQuiz completed! Your score: {self.score}/{self.questions_asked}")
        print(f"Final difficulty level reached: {self.current_difficulty}")

        # Ask user if they want to save results
        save_results = input("\nWould you like to save your results to a CSV file? (y/n): ")
        if save_results.lower().startswith('y'):
            filename = 'results.csv'
            self.save_results_to_csv(filename)


if __name__ == "__main__":
    quiz = quiz_model(sample_data)
    quiz.start_session()





