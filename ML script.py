import pandas as pd 
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import specificity_score



data_path = "./validation_data"

projects = {"ambros" : ["mylyn","pde"],"eclipse" : ["eclipse"], "ck" : ["camel","poi","prop","synapse","xalan","xerces","lucene"]}
projects_features = {"ambros" : ["numberOfVersionsUntil:","numberOfFixesUntil:","numberOfRefactoringsUntil:","numberOfAuthorsUntil:","linesAddedUntil:","maxLinesAddedUntil:","avgLinesAddedUntil:","linesRemovedUntil:","maxLinesRemovedUntil:","avgLinesRemovedUntil:","codeChurnUntil:","maxCodeChurnUntil:","avgCodeChurnUntil:","ageWithRespectTo:","weightedAgeWithRespectTo:"],
            "ck" : ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc"],
            "eclipse" : ["pre","ACD","FOUT_avg","FOUT_max","FOUT_sum","MLOC_avg","MLOC_max","MLOC_sum","NBD_avg","NBD_max","NBD_sum","NOF_avg","NOF_max","NOF_sum","NOI","NOM_avg","NOM_max","NOM_sum","NOT","NSF_avg","NSF_max","NSF_sum","NSM_avg","NSM_max","NSM_sum","PAR_avg","PAR_max","PAR_sum","TLOC","VG_avg","VG_max","VG_sum"]
            }
outcome =  {"ck" : "bug","ambros" : "bugs","eclipse" : "post"}  


df_results = pd.DataFrame(columns = ["project_name","algorithm","file_id","train_or_test","f1","G"])

#main loop
for file_name in os.listdir(data_path) : 
  if "train" in file_name : 
    print("working on : ",file_name)

    models = {"decision_tree_maxdepth_10" : DecisionTreeClassifier(max_depth=10),
          
        
          "gaussian_naive_bayes" :GaussianNB(),
         
          "logistic_regression" :  LogisticRegression(),
          "svm" :  SVC(),
          "random forrest" :  RandomForestClassifier(max_depth=10)
          }

    row = {} 
    train_data = pd.read_csv(data_path+"/"+file_name)
    test_data = pd.read_csv(data_path+"/"+file_name.replace("train","test"))
    project_name = file_name.replace(".csv","").split("_")[0]
    project_id = ""
    row["file_id"] = file_name
    row["project_name"] = project_name
    for project in projects : 
      for pnames in projects[project] : 
        if pnames in project_name :
          project_id = project 
          break
    features = projects_features[project_id]
    output_variable = outcome[project_id]
    for model_id in models : 
      row["algorithm"]= model_id
      model = models[model_id] 
      model.fit(train_data[features],train_data[output_variable])
      
      y_train_predict = model.predict(train_data[features])
      y_test_predict = model.predict(test_data[features])

      row["train_or_test"] = "train"

      row["f1"]= f1_score(train_data[output_variable],y_train_predict)
      row["G"]=geometric_mean_score(train_data[output_variable],y_train_predict)

      df_results = df_results.append(row,ignore_index=True)      
      row["train_or_test"] = "test"

      row["f1"]= f1_score(test_data[output_variable],y_test_predict)
      row["G"]=geometric_mean_score(test_data[output_variable],y_test_predict)

      df_results = df_results.append(row,ignore_index=True)   





   

