import csv
import random
import datetime
from io import StringIO
import string

# Sample data as a fallback option
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

# Medical terminology data for generating random questions
medical_terms = {
    "prefixes": {
        "hypo-": "below normal or deficient",
        "hyper-": "excessive or above normal",
        "tachy-": "rapid or fast",
        "brady-": "slow",
        "dys-": "difficult, painful, or abnormal",
        "a-/an-": "absence or lack of",
        "endo-": "within or inside",
        "exo-": "outside or outward",
        "hemi-": "half",
        "macro-": "large",
        "micro-": "small",
        "neo-": "new",
        "poly-": "many or much",
        "post-": "after or behind",
        "pre-": "before or in front of",
    },
    "suffixes": {
        "-itis": "inflammation",
        "-osis": "condition, usually abnormal",
        "-oma": "tumor or mass",
        "-pathy": "disease",
        "-ectomy": "surgical removal",
        "-otomy": "surgical incision",
        "-plasty": "surgical repair",
        "-scopy": "visual examination",
        "-gram": "record or image",
        "-algia": "pain",
        "-emia": "blood condition",
        "-penia": "deficiency",
        "-megaly": "enlargement",
        "-rrhea": "flow or discharge",
        "-phobia": "fear",
    },
    "organs": {
        "heart": ["pumps blood", "cardiac", "cardiovascular system"],
        "lungs": ["respiration", "pulmonary", "respiratory system"],
        "liver": ["detoxifies blood", "hepatic", "digestive system"],
        "kidneys": ["filter blood", "renal", "urinary system"],
        "brain": ["controls body functions", "cerebral", "nervous system"],
        "stomach": ["digests food", "gastric", "digestive system"],
        "pancreas": ["produces insulin", "pancreatic", "endocrine system"],
        "spleen": ["filters blood", "splenic", "lymphatic system"],
        "thyroid": ["regulates metabolism", "thyroid", "endocrine system"],
        "gallbladder": ["stores bile", "biliary", "digestive system"],
    },
    "medical_terms": {
        "thrombus": "blood clot",
        "embolus": "traveling blood clot",
        "ischemia": "inadequate blood supply",
        "necrosis": "death of tissue",
        "edema": "swelling due to fluid",
        "stenosis": "narrowing of a passage",
        "hypertension": "high blood pressure",
        "hypotension": "low blood pressure",
        "tachycardia": "rapid heart rate",
        "bradycardia": "slow heart rate",
        "dyspnea": "difficulty breathing",
        "apnea": "temporary cessation of breathing",
        "hemoptysis": "coughing up blood",
        "hematemesis": "vomiting blood",
        "hematuria": "blood in urine",
        "melena": "black, tarry stool",
        "jaundice": "yellowing of skin",
        "cyanosis": "bluish discoloration of skin",
        "syncope": "fainting",
        "vertigo": "dizziness",
    },
    "anatomical_terms": {
        "anterior": "front",
        "posterior": "back",
        "superior": "above",
        "inferior": "below",
        "medial": "toward the middle",
        "lateral": "away from the middle",
        "proximal": "closer to the point of attachment",
        "distal": "farther from the point of attachment",
        "superficial": "near the surface",
        "deep": "away from the surface",
        "bilateral": "affecting both sides",
        "unilateral": "affecting one side",
        "ipsilateral": "on the same side",
        "contralateral": "on the opposite side",
        "dorsal": "back side",
        "ventral": "front side",
    },
    "body_systems": {
        "cardiovascular system": ["heart", "blood vessels", "circulation"],
        "respiratory system": ["lungs", "breathing", "gas exchange"],
        "digestive system": ["stomach", "intestines", "digestion"],
        "nervous system": ["brain", "spinal cord", "nerves"],
        "endocrine system": ["hormones", "glands", "regulation"],
        "urinary system": ["kidneys", "bladder", "urine production"],
        "reproductive system": ["gonads", "genitalia", "reproduction"],
        "immune system": ["white blood cells", "antibodies", "defense"],
        "integumentary system": ["skin", "hair", "nails"],
        "musculoskeletal system": ["muscles", "bones", "movement"],
    },
}

def generate_random_questions(num_questions=20):
    """Generate random medical questions with varying difficulty levels"""
    questions = []
    
    # Define question templates with their difficulty levels
    question_templates = [
        # Easy questions (difficulty 0)
        {"template": "What is the medical term for {answer}?", 
         "generator": lambda: (random.choice(list(medical_terms["medical_terms"].items())), 0)},
        {"template": "What does the prefix '{key}' mean in medical terminology?", 
         "generator": lambda: (random.choice(list(medical_terms["prefixes"].items())), 0)},
        {"template": "What does the suffix '{key}' indicate in medical terminology?", 
         "generator": lambda: (random.choice(list(medical_terms["suffixes"].items())), 0)},
        {"template": "What body system includes the {key}?", 
         "generator": lambda: (random.choice([(organ, system) for system, components in medical_terms["body_systems"].items() 
                                             for organ in components if len(organ) > 3]), 0)},
        
        # Medium questions (difficulty 1)
        {"template": "What does the term '{key}' mean in medical context?", 
         "generator": lambda: (random.choice(list(medical_terms["anatomical_terms"].items())), 1)},
        {"template": "What is the primary function of the {key}?", 
         "generator": lambda: ((organ, info[0]) for organ, info in medical_terms["organs"].items()),
         "difficulty": 1},
        {"template": "In which body system would you find the {key}?", 
         "generator": lambda: ((organ, info[2]) for organ, info in medical_terms["organs"].items()),
         "difficulty": 1},
        
        # Hard questions (difficulty 2)
        {"template": "What is the medical term for {description}?", 
         "generator": lambda: ((value, key) for key, value in medical_terms["medical_terms"].items()),
         "difficulty": 2},
        {"template": "Which medical condition is characterized by {answer}?", 
         "generator": lambda: (("excessive " + value, key + "osis") for key, value in random.sample(list(medical_terms["organs"].items()), 5)),
         "difficulty": 2},
        {"template": "What is the anatomical term for {answer}?", 
         "generator": lambda: ((value, key) for key, value in medical_terms["anatomical_terms"].items()),
         "difficulty": 2},
    ]
    
    # Generate questions
    while len(questions) < num_questions:
        # Select a random template
        template = random.choice(question_templates)
        
        try:
            # Get a key-value pair from the generator
            item = next(template["generator"]())
            # Get difficulty from the template
            difficulty = template["difficulty"]
            
            if isinstance(item, tuple):
                key, value = item
            else:
                key, value = item, ""
                
            # Format the question
            question = template["template"].format(key=key, answer=value, description=value)
            
            # Add to questions if not already present
            if not any(q["question"] == question for q in questions):
                questions.append({
                    "question": question,
                    "answer": value if isinstance(value, str) else key,
                    "difficulty": difficulty
                })
        except (StopIteration, TypeError):
            # If generator is exhausted or returns incorrect format, try another template
            continue
    
    # If we couldn't generate enough questions, fill with random ones
    while len(questions) < num_questions:
        difficulty = random.choice([0, 1, 2])
        question = f"Medical question #{len(questions)+1}"
        answer = ''.join(random.choices(string.ascii_uppercase, k=5))
        
        questions.append({
            "question": question,
            "answer": answer,
            "difficulty": difficulty
        })
    
    # Convert to CSV format
    csv_data = "question,answer,score\n"
    for q in questions:
        # Escape quotes in the question and answer
        question = q["question"].replace('"', '""')
        answer = q["answer"].replace('"', '""')
        
        # Add quotes if there are commas in the question or answer
        if ',' in question:
            question = f'"{question}"'
        if ',' in answer:
            answer = f'"{answer}"'
            
        csv_data += f"{question},{answer},{q['difficulty']}\n"
    
    return csv_data


class quiz_model:
    def __init__(self, data=None, use_random=True, num_questions=20):
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

        # Generate random questions if requested
        if use_random:
            data = generate_random_questions(num_questions)
        # Use provided data or fallback to sample data
        elif data is None:
            data = sample_data

        # read csv data
        csv_reader = csv.reader(StringIO(data))
        next(csv_reader)  # skip the header row

        # read the csv files
        for row in csv_reader:
            if len(row) >= 3:  # Ensure row has enough elements
                question, answer, difficulty = row
                # convert the difficulty into an int
                try:
                    difficulty = int(difficulty)
                    # store each question according to difficulty
                    self.question_difficulty[difficulty].append({
                        'question': question,
                        'answer': answer,
                        'difficulty': difficulty
                    })
                except (ValueError, KeyError):
                    # Skip invalid rows
                    continue

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
    # Ask user if they want random questions
    use_random = input("Would you like to use randomly generated questions? (y/n): ").lower().startswith('y')
    
    if use_random:
        num_questions = input("How many questions would you like? (default: 20): ")
        try:
            num_questions = int(num_questions)
        except ValueError:
            num_questions = 20
        
        quiz = quiz_model(use_random=True, num_questions=num_questions)
    else:
        quiz = quiz_model(sample_data, use_random=False)
    
    quiz.start_session()





