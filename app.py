from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Gemini API
def initialize_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.start_chat(history=[])

@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    data = request.json
    name = data.get('name', '')
    role = data.get('role', '')
    level = data.get('level', '')
    experience = data.get('experience', '')
    num_questions = data.get('num_questions', 3)

    chat = initialize_gemini()

    question_prompt = f""" 
    Act as a technical interviewer. Based on the following:
    - Role: {role}
    - Level: {level}
    - Candidate Experience: {experience}

    Generate {num_questions} realistic technical interview questions. 
    Return them in a numbered list format like: "1. Question text"
    Do NOT include answers or feedback yet.
    """

    try:
        response = chat.send_message(question_prompt)
        questions = [q.strip() for q in response.text.split("\n") if q.strip() and any(c.isdigit() for c in q)]

        session_id = os.urandom(16).hex()

        return jsonify({
            'success': True,
            'session_id': session_id,
            'questions': questions[:num_questions],
            'chat_state': response.candidates[0].content.parts[0].text  # Optional
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/evaluate-answers', methods=['POST'])
def evaluate_answers():
    data = request.json
    name = data.get('name', '')
    role = data.get('role', '')
    level = data.get('level', '')
    experience = data.get('experience', '')
    qa_pairs = data.get('qa_pairs', [])

    chat = initialize_gemini()

    formatted_qa = "\n".join([f"Q: {pair['question']}\nA: {pair['answer']}" for pair in qa_pairs])

    evaluation_prompt = f"""
    Act as a senior interviewer. Based on the following interview:

    Candidate: {name}
    Role: {role}
    Level: {level}
    Experience: {experience}

    Questions and Answers:
    {formatted_qa}

    Now provide:
    1. A score out of 10
    2. A brief summary of strengths
    3. Areas for improvement
    4. One piece of personalized advice

    Format your response clearly with these sections:
    - Overall Score
    - Strengths
    - Areas for Improvement
    - Personalized Advice
    """

    try:
        final_feedback = chat.send_message(evaluation_prompt)

        return jsonify({
            'success': True,
            'evaluation': final_feedback.text
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
