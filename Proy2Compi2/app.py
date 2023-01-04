from flask import Flask, render_template,request,send_from_directory,Response
import os
import io
#desicion tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.datasets import make_regression
from sklearn.gaussian_process.kernels import RBF
#n3uronales
#
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

#gaus
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
#gaus
#regresion lineado
#ini
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, precision_score,classification_report
from sklearn.preprocessing import PolynomialFeatures
#fin
##ima ini
import base64
##ima fin
import pandas as pd
import numpy  as np
import streamlit as st
from matplotlib import pyplot as plot
##from matplotlib.figure import Figurecls

##graficar
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')
from flask import Flask
import numpy as np
plot.rcParams["figure.figsize"] = [7.50, 3.50]
plot.rcParams["figure.autolayout"] = True


#archivos permitidos
ALLOWED_EXTENSIONS = set(["csv","xml","xlsx","json"])

def permitirExtension(filename):
    return "." in filename and filename.rsplit(".",1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)

@app.route('/',methods=["GET"])
def index():
   return render_template('index.html')

@app.route('/', methods=["POST"])
def predict():
 if request.method == 'POST':
    file =request.files['file']
    file_phat = "./archivos/"+file.filename
    if file and permitirExtension(file.filename):  
        file.save(file_phat)
        tipo_ext = file.filename.rsplit(".", 1)[1]
        if tipo_ext == "csv":  #para todos los metodos
            # dcide el metodo a ejecutar
            z = request.form.get('datalistOptions')
            if z == "1": #Regresion Lineal
                x_name = request.form['x_name']
                y_name = request.form['y_name']
                prediccionL = request.form['prediccion']
                df = pd.read_csv(file_phat)  # read_csv
                return RegresionLineal(x_name, y_name, df,prediccionL)
           # return RegresionLineal(x_name, y_name, df)
            elif z == "2": #Regresion Polinomial
                x_namep = request.form['x_namep']
                y_namep = request.form['y_namep']
                pred = request.form['Prediccion']
                grado = request.form['grado']
                dfp = pd.read_csv(file_phat)  # read_csv
                return RegresionPolinomial(x_namep, y_namep, dfp, int(grado),pred)
            elif z == "3":  # Clasification Gauss
                #dataset = pd.read_csv(file_phat)  # read_csv
                df = pd.read_csv(file_phat)  # read_csv
                return GaussClasificators(df)
                #return "Clasificador Gaussiano"
            elif z == "4":  # Tree Desitions
                df = pd.read_csv(file_phat)  # read_cs
                ex = request.form['explicativas']
                number = request.form['number']
                return TreeDesitions(df,ex,number)
            elif z == "5":  # Neuronal Network
                df = pd.read_csv(file_phat)  # read_cs
                x_namen = request.form['x_namen']
                y_namen = request.form['y_namen']
                valPredecir = request.form['pred']
                
                #entranamiento de datos
                return RedesNeuronales(df,x_namen,y_namen,valPredecir)
                
                 
                 
    
        return "NO se ha ingresado archivo"
    #REGRESION Lineal
def RegresionLineal (x_name,y_name,df,prediccion):
#INICIOs
    x = df[x_name].values.reshape(-1, 1)  # Level
    y = df[y_name].values.reshape(-1, 1)  # Salary
    plot_url = plot.scatter(x, y)
    #plot.plot(x, y)
    #plot.scatter(x, y)
    model = LinearRegression()
    model.fit(x, y)
    mlr = model
    y_pred = model.predict(x)  ## mas cercano a 1 mas precisas las predicciones.
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    ########################
    
    y_pred2 = mlr.predict([[int(prediccion)]])  
    
    #######################
    plot.plot(x, y_pred,color='r')
    img = io.BytesIO()
    plot.savefig(img, format='png')
    plot.clf()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template('Analyze.html', valor1=rmse, valor2=r2, valor3= y_pred2 ,url=plot_url)
    #FIN

    #Regresion Polinomial

def RegresionPolinomial(x_namep,y_namep,dfp,grado,pred):
    xp = dfp[x_namep].values.reshape(-1, 1)  # 
    yp = dfp[y_namep].values.reshape(-1, 1)  # 
    polinomial = PolynomialFeatures(degree=int(grado), include_bias=False)
    xpoly = polinomial.fit_transform(xp)
    modelp = LinearRegression()
    modelp.fit(xpoly,yp)
    y_predp= modelp.predict(xpoly)
    modelaux = modelp
    y_predI =modelaux.predict([[int(pred),0]])
    plot.scatter(xp,yp)
    plot.plot(xp,y_predp, color="r")
    rmse= np.sqrt(mean_squared_error(yp,y_predp))
    r2 = r2_score(yp,y_predp)
    imgp = io.BytesIO()
    plot.savefig(imgp, format='png')
    plot.clf()
    imgp.seek(0)
    plot_urlp = base64.b64encode(imgp.getvalue()).decode()
    return render_template('Analyze2.html', valor1=rmse, valor2=r2, valor3= y_predI , urlp=plot_urlp)

def GaussClasificators(dataset):  # dataset
   #url = path  # 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
   #raw_data = urllib.request.urlopen(url)
   #dataset = np.loadtxt(raw_data, delimiter=',')
   x = dataset.iloc[:,0:48].values
   y = dataset.iloc[:, -1].values
   x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=17)
   
  # print(x_train)
  # print(x_test)
  # print(y_train)
  # print(y_test)   
   ####
   ##kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
   ##gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
   ##gaussian_process.fit(x_train, y_train)
   ##gaussian_process.kernel_
   ##mean_prediction, std_prediction = gaussian_process.predict(x, return_std=True)
   ##prediccion
   
   ####

   BernNB = BernoulliNB(binarize=True)
   BernNB.fit(x_train, y_train)
   y_expect = y_test # y_test
   y_pred = BernNB.predict(x_test)
   #print(accuracy_score(y_expect, y_pred)) ## puntuacion d presiicion 
   GausNB = GaussianNB() 
   GausNB.fit(x_train, y_train)
   y_pred = GausNB.predict(x_train) #x_test
   
   #valor2 = gpc.score(x, y)
   #GNBclf = GaussianNB()
   #GNBclf.fit(x, y)
   #kernel = 1.0 * RBF(1.0)
   #gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(x, y)
   #gpc.score(x,y)
    #METRICAS
   clf = MultinomialNB(force_alpha=True)
   clf.fit(x, y)
   MultinomialNB(force_alpha=True)
   print("",clf.predict(x[2:3]))
   plot.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='RdBu')
   l = plot.axis()
   plot.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=20, cmap='RdBu', alpha=0.1)
   plot.axis(l)
   imgp = io.BytesIO()
   plot.savefig(imgp, format='png')
   plot.clf()
   imgp.seek(0)
   plot_urlp = base64.b64encode(imgp.getvalue()).decode()    
   return render_template('Analyze3.html', valor1=accuracy_score(y_expect, y_pred), valor2=recall_score(y_expect, y_pred, average='micro'), valor3=precision_score(y_expect, y_pred, average='micro'), urlp=plot_urlp)



def TreeDesitions(dataset,explicativas,number):
    #converir columnas en numeros
   
   dataset = pd.get_dummies(data=dataset, drop_first=True)
   x = dataset.drop(dataset.columns[int(explicativas)], axis=1)
   y = dataset[dataset.columns[int(explicativas)]]
   X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
   modelo = DecisionTreeClassifier()
   #modelo.fit(x, y)
   modelo.fit(X=x,y=y)
   #predicciones
   if number!="":
       a= x.sample(int(number))
       prob =modelo.predict_proba(a)
       a= a.to_html()
   else:
       a='No existe Informacion para prediccion'
       prob=0
    #   a="No existe informacion para Predecir" 
   y_pred= modelo.predict(X_train)# prediccion de las explicativas
   acuarince = accuracy_score(y_train,y_pred)
   clasificator = classification_report(y_train,y_pred)
   
   #resultado = a
   ######end
   
       
   plot_tree(decision_tree=modelo, feature_names=x.columns,filled=True)
   imgp = io.BytesIO()
   plot.savefig(imgp, format='png')
   plot.clf()
   imgp.seek(0)
   plot_urlp = base64.b64encode(imgp.getvalue()).decode()
   return render_template('Analyze4.html',  valor1 = a ,valor2=clasificator, valor3 =acuarince ,urlp=plot_urlp)

   
   
   #dataset = pd.get_dummies(data=dataset, drop_first=True)
  
   #print(dataset.columns)
   ##explicativas = dataset.drop(columns='outlook')
   ##objetivas = dataset.outlook
  ## model = DecisionTreeClassifier()
  # model.fit(x=explicativas, y=objetivas)
   #plot_tree(decision_tree=model,feature_names=dataset.columns,filled=True)
  ## imgp = io.BytesIO()
  ## plot.savefig(imgp, format='png')
  ## plot.clf()
  ## imgp.seek(0)
  ## plot_urlp = base64.b64encode(imgp.getvalue()).decode()
  ## return render_template('Analyze3.html',  urlp=plot_urlp)


   #seleeccion automatizada de variables
   #explicativas y objetivas
def RedesNeuronales(df,namex,namey,pred):
    
   # x= df[df.columns[0]] ## tiemo
   # y= df[df.columns[1]] ## carga
    
    x = df[df.columns[int(namex)]].values.reshape(-1, 1)  # Level
    y = df[df.columns[int(namey)]].values.reshape(-1, 1)  # Salary

    #regresion
    X=x[:,np.newaxis]
    X_train,X_test,y_train,y_test = train_test_split(x,y)

    #prediccion
    mlr = MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(3,3),random_state=1)
    #entrenar los datos
    mlr.fit(X_train,y_train)
    #score para todo el modelo
    Sscore=mlr.score(X_train,y_train)
    #prediccion
    
    if pred != "" :
        y_pred = mlr.predict([[int(pred)]])  
    else:
        y_pred= "No existe valor para prediccion"
     #  graficar iteracions 
      
    i =0
    plot.figure(figsize=(20,10))
    colores=['teal','pink','brown','black','gray','silver','red','tan','blue','plum','gold','gold','orange','yellow','navy','olive','aqua','indigo','cyan','lime','green']
    while True: 
        i=i+1
        X_train,X_test,y_train,y_test = train_test_split(x,y)
        mlr = MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(3,3),random_state=1)
        mlr.fit(X_train,y_train)
        plot.scatter(X_train,y_train,color='red',label='Training set' if i==1 else '')
        plot.scatter(X_test,y_test,color='blue',label='Test set' if i==1 else '')
        plot.scatter(X,mlr.predict(x),color=colores[i],label='Iteracion set' + str(i))
        if mlr.score(X_train,y_train)>0.98 or i==20: # valor Modelo graficado.
            break
    plot.xlabel('Eje X')
    plot.ylabel('Eje Y')
    plot.legend(loc="upper right")
    
    imgp = io.BytesIO()
    plot.savefig(imgp, format='png')
    plot.clf()
    imgp.seek(0)
    plot_urlp = base64.b64encode(imgp.getvalue()).decode()    
   
   #Mlregresor:
   
    X, y = make_regression(n_samples=int(pred), random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=1)
    regr = MLPRegressor(random_state=1, max_iter=int(pred)).fit(X_train, y_train)
    regr.predict(X_test[:2])
    regr.score(X_test, y_test)

    return render_template('Analyze5.html', valor3=regr.predict(X_test[:2]), valor4=regr.score(X_test, y_test), valor1=Sscore, valor2=y_pred,  urlp=plot_urlp)
            
if __name__ == '__main__':
    # modo depuarcion igual a true.
    app.run(host='0.0.0.0', port=4000, debug=True)
