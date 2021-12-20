from flask import Flask,request,jsonify
import numpy as np
import pickle

app=Flask(__name__)
model = pickle.load(open('student_model.pkl','rb'))

@app.route('/predict',methods=['POST'])
def predict_placement():
    # cgpa = request.json['cgpa']
    # gender=request.json['gender']
    # if gender=='Male':
    #     cgpa=5.0
    # else:
    #     cgpa=8.0
    cgpa=request.json['cgpa']
    iq = int(request.json['iq'])
    profile_score = int(request.json['profile_score'])

    # prediction
    result = model.predict(np.array([cgpa,iq,profile_score]).reshape(1,3))
    print(result)
    
    if result[0] == 1:
            result = 'placed'
    else:
        result = 'not placed'
        
    return jsonify({
            'Prediction': result
        })


if __name__=="__main__":
    app.run(debug=True)