from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import re

load_dotenv()

app = Flask(__name__)

chat_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

SYSTEM_PROMPT = """You are a helpful, knowledgeable, and concise AI assistant.
Answer questions clearly and accurately. Use markdown formatting such as **bold**,
*italic*, `code`, and bullet points where appropriate.
When writing math, use LaTeX with \\( ... \\) for inline and \\[ ... \\] for display equations.
Important LaTeX rules:
- Every \\left( must have a matching \\right)
- Every \\left[ must have a matching \\right]
- Every \\left\\{ must have a matching \\right\\}
- Never leave a \\left or \\right without its matching pair
- If you don't need stretchy delimiters, just use plain ( ) [ ] { } without \\left/\\right"""

chat_history = []


def fix_latex(text: str) -> str:
    """
    LaTeX cleanup:
    1. Remove literal MathJax error strings echoed by the LLM.
    2. Normalise $$ ... $$ to \\[ ... \\]
    3. Normalise $ ... $ to \\( ... \\) (only when content looks like math)
    4. Balance unmatched \\left / \\right pairs inside each math block.
    """

    # ── 1. Remove literal MathJax error strings ────────────────────────────
    text = re.sub(r'Missing \\left or extra \\right\.?', '', text)
    text = re.sub(r'Missing or unrecognized delimiter for \\right\.?', '', text)
    text = re.sub(r'\\leftMissing or unrecognized delimiter for \\right\.?', '', text)

    # ── 2. Normalise $$ ... $$ → \[ ... \] ─────────────────────────────────
    text = re.sub(r'\$\$([\s\S]*?)\$\$', r'\\[\1\\]', text)

    # ── 3. Normalise $ ... $ → \( ... \) ───────────────────────────────────
    text = re.sub(
        r'\$([^$\n]+?)\$',
        lambda m: f'\\({m.group(1)}\\)' if re.search(r'[\\^_{}]|\\[a-zA-Z]', m.group(1)) else m.group(0),
        text
    )

    # ── 4. Balance \left / \right inside each math block ───────────────────
    PAIRS = {'(': ')', '[': ']', '{': '}', '|': '|', '.': '.'}

    def balance_block(math: str) -> str:
        """Add missing \\right<x> for every unmatched \\left<x>."""
        stack = []
        for delim in re.findall(r'\\left\s*([(\[{|.])', math):
            stack.append(PAIRS.get(delim, '.'))
        for _ in re.findall(r'\\right\s*([)\]}|.])', math):
            if stack:
                stack.pop()
        for closer in reversed(stack):
            math = math + f'\\right{closer}'
        return math

    text = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: '\\[' + balance_block(m.group(1)) + '\\]', text)
    text = re.sub(r'\\\(([\s\S]*?)\\\)', lambda m: '\\(' + balance_block(m.group(1)) + '\\)', text)

    return text


@app.route('/')
def index():
    return render_template('index_chatbot.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for entry in chat_history:
        if entry['role'] == 'user':
            messages.append(HumanMessage(content=entry['content'])) # type: ignore
        else:
            messages.append(AIMessage(content=entry['content'])) # type: ignore
    messages.append(HumanMessage(content=user_message)) # type: ignore

    raw_reply = chat_model.invoke(messages).content
    clean_reply = fix_latex(raw_reply) # type: ignore

    chat_history.append({'role': 'user',      'content': user_message})
    chat_history.append({'role': 'assistant', 'content': clean_reply})

    return jsonify({'reply': clean_reply})


@app.route('/clear', methods=['POST'])
def clear():
    chat_history.clear()
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    app.run(debug=True)