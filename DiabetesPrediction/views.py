from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Edit Newly
import numpy as np
sns.set()
from mlxtend.plotting import plot_decision_regions
import missingno as msno
from pandas.plotting import scatter_matrix  # Fixed import statement
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # Assuming you're using KNN for plot_decision_regions
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score  # Fixed import statement
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from io import BytesIO
import base64
#%matplotlib inline

#End 






from django.http import HttpResponse
def home(request):
    return render(request,'home.html')

def predict(request):
    return render(request,'predict.html')

def result(request):
    data = pd.read_csv(r'D:\NLP\DiabetesPrediction\DiabetesPrediction\templates\diabetes.csv')
    x = data.drop('Outcome',axis=1)
    y = data['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2)


    model = LogisticRegression()
    model.fit(x_train,y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])

    result1 = " "
    if pred==[1]:
        result1 = 'Posivite'

    else:
        result1 = 'Negetive'
    

    return render(request,'predict.html',{"result2":result1})
# ,{"result2":result1}

def accuracy(request):
    df = pd.read_csv(r'D:\NLP\DiabetesPrediction\DiabetesPrediction\templates\diabetes.csv')
    tableshow = df.columns
    nullcheck = df.isnull().sum()

    # aiming to impute NAN values for the columns in accordance with their distribution
    df_copy = df.copy(deep=True)
    df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

    columns_to_fill = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for column in columns_to_fill:
         df_copy[column].fillna(df_copy[column].mean(), inplace=True)

    # Plotting NULL Count Analysis Plot
    p = msno.bar(df, figsize=(8, 6))  # Adjust the figsize parameter here
    buffer = BytesIO()
    p.get_figure().savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html = f'<img src="data:image/png;base64,{plot_data_uri}"/>'

    # not necessary
    # Checking the balance of the data by plotting the count of outcome by their values
    colour_wheel = {1: "#0392cf", 2: "#7bc043"}
    colors = df['Outcome'].map(lambda x: colour_wheel.get(x+1))
    print(df.Outcome.value_counts())
    p = df.Outcome.value_counts().plot(kind='bar')
    
    # not necessary
    # Pairplot with smaller size
    pairplot = sns.pairplot(df, hue='Outcome', height=3)  # Adjust the height parameter here
    buffer_pairplot = BytesIO()
    pairplot.savefig(buffer_pairplot, format='png')
    buffer_pairplot.seek(0)
    plot_data_uri_pairplot = base64.b64encode(buffer_pairplot.read()).decode('utf-8')
    plot_html_pairplot = f'<img src="data:image/png;base64,{plot_data_uri_pairplot}"/>'

    return render(request, 'accuracy.html', {'accuracy': tableshow, 'nullcheck': nullcheck, 'plot_html': plot_html, 'plot_html_pairplot': plot_html_pairplot})


def correlation(request):
    
    df = pd.read_csv(r'D:\NLP\DiabetesPrediction\DiabetesPrediction\templates\diabetes.csv')
    correlation_matrix = df.corr()

    # Plotting the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn')

    # Save the plot to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html = f'<img src="data:image/png;base64,{plot_data_uri}" alt="Correlation Heatmap"/>'


    df_copy = df.copy(deep=True)
    # df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

    # Correlation between all the features After clearing
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_copy.corr(), annot=True, cmap='RdYlGn')

    # Save the plot to a BytesIO buffer
    buffer_after = BytesIO()
    plt.savefig(buffer_after, format='png')
    buffer_after.seek(0)
    plot_data_uri_after = base64.b64encode(buffer_after.read()).decode('utf-8')
    plot_html_after = f'<img src="data:image/png;base64,{plot_data_uri_after}" alt="Correlation Heatmap after clearing"/>'

    # extra
    sc_X = StandardScaler()
    x = pd.DataFrame(sc_X.fit_transform(df_copy.drop(['Outcome'],axis=1),), columns=['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction','Age'])
    x.head()
    # extra
    y = df_copy.Outcome

 # K-Nearest Neighbor (KNN)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    test_scores = []
    train_scores = []

    for i in range(1, 15):
        knn = KNeighborsClassifier(i)
        knn.fit(x_train, y_train)
        train_scores.append(knn.score(x_train, y_train))
        test_scores.append(knn.score(x_test, y_test))

    max_train_score = max(train_scores)
    train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

    max_test_score = max(test_scores)
    test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

#Knn test Score
    knn = KNeighborsClassifier(11)
    knn.fit(x_train, y_train)
    knn_test_score = knn.score(x_test, y_test)



    # Plot decision regions
    value = 20000
    width = 20000
    buffer_decision_regions = BytesIO()
    plt.figure(figsize=(6, 6))  # Adjust the figsize parameter to your desired size
    plot_decision_regions(x.values, y.values, clf=knn, legend=2,
                        filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value},
                        filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width},
                        X_highlight=x_test.values, ax=plt.gca())
    plt.title('KNN with Diabetes Data')
    plt.savefig(buffer_decision_regions, format='png')
    buffer_decision_regions.seek(0)
    knn_decision_regions_plot = base64.b64encode(buffer_decision_regions.read()).decode('utf-8')

#Confusion Matrix
   #Confusion Matrix
    y_pred = knn.predict(x_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    # Save the plot to a BytesIO buffer
    buffer_cnf_matrix = BytesIO()
    plt.savefig(buffer_cnf_matrix, format='png')
    buffer_cnf_matrix.seek(0)
    plot_data_uri_cnf_matrix = base64.b64encode(buffer_cnf_matrix.read()).decode('utf-8')
    plot_html_cnf_matrix = f'<img src="data:image/png;base64,{plot_data_uri_cnf_matrix}" alt="Confusion Matrix"/>'

    # ...
   
    param_grid = {'n_neighbors': np.arange(1, 15)}
    knn = KNeighborsClassifier()
    knn_cv = GridSearchCV(knn, param_grid, cv=5)
    knn_cv.fit(x_train, y_train)

    best_k = knn_cv.best_params_['n_neighbors']
    best_score = knn_cv.best_score_

    # Pass the max train, test scores, decision regions plot, confusion matrix plot, and best parameters to the template
    return render(request, 'correlation.html', {'plot_html_before': plot_html, 
                                                'plot_html_after': plot_html_after, 
                                                'max_train_score': max_train_score * 100, 
                                                'max_test_score': max_test_score * 100, 
                                                'knn_test_score': knn_test_score,
                                                'knn_decision_regions_plot': knn_decision_regions_plot,
                                                'confusion_matrix_plot': plot_html_cnf_matrix,
                                                'best_k': best_k,
                                                'best_score': best_score})