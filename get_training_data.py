import csv
import datetime
import random
import time
import os

class get_training_data:
    def __init__(self):
        self.question_bank = [
            {
                'question': "Which scalpel blade is primarily used for making large skin incisions, such as in laparotomy?",
                'answer': "#10 blade",
                'difficulty': 0},

            {'question': "What type of scissors are straight and used for cutting sutures?",
             'answer': "Straight Mayo scissors",
             'difficulty': 0},

            {
                'question': "Which forceps are toothed at the tip and used for handling dense tissue, such as in skin closures?",
                'answer': "Adson forceps",
                'difficulty': 0},

            {'question': "What is the primary use of Metzenbaum scissors in surgical procedures?",
             'answer': "Cutting delicate tissue and for blunt dissection",
             'difficulty': 1},

            {
                'question': "Which clamp is described as a traumatic toothed instrument used to hold tissue that will be removed?",
                'answer': "Kocher clamp",
                'difficulty': 1},

            {'question': "What is the function of a Crile hemostat during surgery?",
             'answer': "grasp tissue or vessels that will be tied off",
             'difficulty': 2},

            {'question': "Describe the differences between a tapered needle and a conventional cutting needle.",
             'answer': "Tapered needles are round and used in soft tissue, cutting needles are triangular with a sharp inner edge, used for tougher tissue like skin",
             'difficulty': 2},

            {'question': "Explain the significance of suture sizing.",
             'answer': "Higher numbers indicate larger diameters, more zeros indicate smaller diameters",
             'difficulty': 1},

            {'question': "Orthopedics is the branch of surgery connected with conditions involving what system?",
             'answer': "Musculoskeletal system)",
             'difficulty': 1},

            {'question': "What are the considerations for choosing between braided and monofilament sutures?",
             'answer': "Braided sutures provide better knot security, monofilament causes less tissue drag and is less prone to infection",
             'difficulty': 2},

            {
                'question': "A patient has undergone ACL reconstruction surgery. What instruments shown in the module could be used on this operation?",
                'answer': "sutures, scalpel, scissors, clamps",
                'difficulty': 0},

            {'question': "Which specialty surgery category includes information about the thyroid gland?",
             'answer': "Endocrine Surgery",
             'difficulty': 0},

            {
                "question": "What is the common use for a #15 scalpel blade?",
                "answer": "For making finer incisions",
                "difficulty": 0
            },
            {
                "question": "What type of needle is most commonly used in soft tissue like the intestine?",
                "answer": "Tapered needle",
                "difficulty": 1
            },
            {
                "question": "Which suction tip is commonly used in ENT and neurosurgery and is usually angled?",
                "answer": "Frazier suction tip",
                "difficulty": 1
            },
            {
                "question": "Which instrument is used to gain exposure of skin layers?",
                "answer": "Army-Navy retractor",
                "difficulty": 0
            },
            {
                "question": "What type of suture material is Prolene® classified as?",
                "answer": "Non-absorbable monofilament",
                "difficulty": 2
            }
        ]

        self.avg_response_time = 0
        self.consecutive_correct = 0
        self.topic_expertise = 0
        self.quiz_results = []

    def run_quiz(self):
        # Shuffle the questions in random order
        random.shuffle(self.question_bank)
        total_correct = 0

        for question in self.question_bank:
            # Ask the user the question
            print(f"Question: {question['question']}")
            start_time = time.time()  # Start the timer

            user_answer = input("Your answer: ").strip()  # Take the user's answer
            end_time = time.time()  # Stop the timer

            # Calculate response time
            response_time = end_time - start_time
            self.avg_response_time = (self.avg_response_time + response_time) / 2  # Update average response time

            is_correct = (user_answer == question['answer'])
            question_answered_correctly = 0
            if is_correct:
                print(f"correct! ")
                question_answered_correctly = 1
                total_correct += 1
                self.consecutive_correct += 1
            else:
                print("Incorrect!")
                question_answered_correctly = 0
                if question_answered_correctly == 0:
                    question_answered_correctly = 0
                self.consecutive_correct = 0
            user_score = total_correct / len(self.question_bank)

            result = {
                'question': question['question'],
                'answer': question['answer'],
                'difficulty': question['difficulty'],
                'correct': question_answered_correctly,
                'time_taken': response_time,
                'score': user_score
            }
            self.quiz_results.append(result)
            # Give feedback on the user's consecutive correct answers
            print(f"Consecutive correct answers: {self.consecutive_correct}\n")

        print("Quiz completed.")

        print(f"Score: {user_score}")
        print(f"Average response time: {self.avg_response_time:.2f} seconds")
        print(f"Final streak of consecutive correct answers: {self.consecutive_correct}")

    # def check_answer(self, user_answer, correct_answer, threshold=0.7):
    #     vectorizer = TfidfVectorizer(stop_words='english')
    #     vectors = vectorizer.fit_transform([user_answer.lower()], [correct_answer.lower()])
    #
    #     similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    #
    #     return similarity, similarity >= threshold

    def save_to_csv(self, filename):
        try:
            results_dir = "user_results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            file_path = os.path.join(results_dir, filename)
            with open(filename, 'w') as csvfile:
                fieldnames = [
                    'question',
                    'answer',
                    'difficulty',
                    'correct',
                    'time_taken',
                    'score'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # write the record to a csv
                for result in self.quiz_results:
                    writer.writerow(result)  # write the row with the quiz results

                print(f"quiz saved to {filename}")
                return True
        except Exception as e:
            print(f"Error saving csv results: {e}")
            return False


if __name__ == "__main__":
    quiz = get_training_data()
    quiz.run_quiz()

    current_date = str(datetime.date.today())
    username = input("Please enter your name/initials: ")

    filename = username + "_quiz_results_" + current_date + ".csv"

    # Create user_results directory if it doesn't exist
    results_dir = "user_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Check if file exists in the user_results directory
    counter = 1
    while os.path.exists(os.path.join(results_dir, filename)):
        filename = f"{username}_quiz_results_{current_date}_{counter}.csv"
        counter += 1

    quiz.save_to_csv(filename)
