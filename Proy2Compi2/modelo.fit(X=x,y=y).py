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
   clasificator = classification_report(y_test,y_pred)
   
   #resultado = a
   ######end
   
       
   plot_tree(decision_tree=modelo, feature_names=x.columns,filled=True)
   imgp = io.BytesIO()
   plot.savefig(imgp, format='png')
   plot.clf()
   imgp.seek(0)
   plot_urlp = base64.b64encode(imgp.getvalue()).decode()
   return render_template('Analyze4.html',  valor1 = a ,valor2=clasificator, valor3 =acuarince ,urlp=plot_urlp)

   