FROM  python:3.10-alpine

WORKDIR /server
COPY . .


RUN /usr/local/bin/python -m pip install  --upgrade pip
#ADD requisitos.txt .
#RUN pip install -r requisitos.txt
RUN pip install flask
RUN pip install flask-cors
#RUN pip install sklearn
RUN pip install scikit-learn
#RUN pip install pandas
RUN pip install waitress


EXPOSE 5000

CMD ["python","app.py"]

#pip install scikit-learn
#pandas
#streamlit
#numpy
#matplotlin
#flask
#flas-cors

#parametros por defecto
#leve,salary
#imagen en base64 y la mandas al front
#El rmse no se pide en el enunciado verdad?
#regreson polinomial
#parametros: regresion: Level,Salary
#datos.key : encabezado
#pasar como parametro
# si prediccion


#POLINOMIAL::::
##parametros quemados.
#un cambiante podrriamos dejar quemado.. 2.
#Data Set habira que tener... cuanto necesarios uno por cada cosa.
#laberinto del falken..
# si prediccion
#Gaussioano.
#sino mal recuerdo el Gaussiano y la redes neuronales devuelven solo datos, no se pueden graficar, de ahi las lineales, polinomiales y arboles si
