from flask import Flask, render_template, request
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
import os

load_dotenv()
app = Flask(__name__)

# Initialize the model and prompt template once
model = ChatOpenAI()
# Get the absolute path to the directory where app.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to your JSON file
template_path = os.path.join(base_dir, 'template.json')

# Load using the full path
template = load_prompt(template_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    # Default values for the form
    paper_input = "Attention Is All You Need"
    style_input = "Beginner-Friendly"
    length_input = "Short (1-2 paragraphs)"

    if request.method == 'POST':
        # Get data from the submitted form
        paper_input = request.form.get('paper_input')
        style_input = request.form.get('style_input')
        length_input = request.form.get('length_input')

        # Run the LangChain logic
        chain = template | model
        result = chain.invoke({
            'paper_input': paper_input,
            'style_input': style_input,
            'length_input': length_input
        })
        summary = result.content

    return render_template('index.html', 
                           summary=summary, 
                           paper_input=paper_input, 
                           style_input=style_input, 
                           length_input=length_input)

if __name__ == '__main__':
    app.run(debug=True)
