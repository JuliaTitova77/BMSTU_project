from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

def get_prediction(params_float):
    pickle_model_file = './Models/kn_model.pkl'
    with open(pickle_model_file, 'rb') as file:
        pickle_model = pickle.load(file)
    Ypredict_pckl = pickle_model.predict(params_float)
    return Ypredict_pckl[0]

@app.route('/', methods=['post', 'get'])
def processing():
    depth = 0.0
    width = 0.0
    message = 'test'
    if request.method == 'POST':
        IW = request.form.get('IW')
        IF = request.form.get('IF')
        VW = request.form.get('VW')
        FP = request.form.get('FP')
        params = [IW,IF,VW,FP]
        print(params)
        try:
            params_float = np.array([[float(param) for param in params]])
            depth, width = get_prediction(params_float)
            message = f"depth: {depth}, width: {width}"
        except Exception as e:
            print(e)
    print(message)
    return render_template('index.html', message = message)

if __name__ == '__main__':
    app.run()