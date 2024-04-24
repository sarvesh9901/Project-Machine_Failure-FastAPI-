from pydantic import BaseModel
import uvicorn
import pickle

pickle_in = open("logistic.pkl","rb")
classifier=pickle.load(pickle_in)
class Model(BaseModel):
    Rotational_speed_rpm:int
    Torque_Nm:float
    Tool_wear_min:int
    TWF:int
    HDF:int
    PWF:int
    OSF:int

from fastapi import FastAPI , requests

app = FastAPI()
@app.get('/')
def hello():
    return {"Welcome "}
@app.get('/welcome')
def test(name:str):
    return {f"Hello {name}"}


@app.post('/predict')
def predict(data:Model):
    data = data.dict()
    Rotational_speed_rpm = data['Rotational_speed_rpm']
    print(Rotational_speed_rpm)
    Torque_Nm =  data['Torque_Nm']
    Tool_wear_min = data['Tool_wear_min']
    TWF = data['TWF']
    HDF = data['HDF']
    PWF = data['PWF']
    OSF = data['OSF']
    prediction = classifier.predict([[Rotational_speed_rpm, Torque_Nm,Tool_wear_min,TWF,HDF,PWF,OSF]])
    return{'prediction':prediction}
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)