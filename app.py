from flask import Flask
from flask import request, jsonify
import requests
from ctransformers import AutoModelForCausalLM


app = Flask(__name__)


# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("Pasindu751/genify-llama2-q8_0", model_file="myllama-7b-v0.1.gguf", model_type="llama", gpu_layers=0,max_new_tokens=128)


@app.route('/', methods=['GET'])
def get_documentation():
        return jsonify({"documentation": "hello"})


@app.route('/generate', methods=['POST'])
def summarize():
    if request.method == 'POST':
        try:
            data = request.json
            input_text = data.get('input_text')
            if input_text:
                output = llm(input_text)
                return jsonify(output), 200
            else:
                return jsonify({'error': 'Input text is required.'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
