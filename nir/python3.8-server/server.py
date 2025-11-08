from flask import Flask, request, jsonify
import spacy
import coreferee

app = Flask(__name__)

# Загружаем модель (coreferee требует старую spacy)
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('coreferee')

@app.route('/resolve', methods=['POST'])
def resolve_coref():
    """
    Принимает JSON: { "text": "...", "chunk_id": "..." }
    Возвращает сущности с учетом кореференций.
    """
    data = request.get_json()
    text = data.get('text', '')
    chunk_id = data.get('chunk_id', '0')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    doc = nlp(text)

    entities = []
    for chain in doc._.coref_chains:
        main = chain.main.text
        mentions = [m.text for m in chain]
        entities.append({
            "main": main,
            "mentions": mentions,
            "chunk_id": chunk_id
        })

    return jsonify({'entities': entities})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8008)
