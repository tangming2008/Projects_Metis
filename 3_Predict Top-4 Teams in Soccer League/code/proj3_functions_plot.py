# function for Project 3

from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import itertools

from sklearn import metrics, cross_validation
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.learning_curve import learning_curve

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

def save_fig(fig, func):
    '''
    Define a function to save high-resolution images
    '''
    fig.savefig('/Users/MingTang/Desktop/figure.jpg', format='jpg', dpi=1000)
    # fig = ax.get_figure()
    # fig.savefig('/Users/MingTang/Desktop/figure.jpg', format='jpg', dpi=1000)
    # save_fig(plt.figure(figsize=(12,7)), plot_learning_curve_models(models, models_name, X, y))

def correct_teamname(teamname):
    correct_teamnames = ['Alaves', 'Athletic Bilbao', 'Atletico Madrid', 'Barcelona', 'Betis', 'Celta Vigo', 'Deportivo La Coruna',
     'Eibar', 'Espanyol', 'Granada', 'Las Palmas', 'Leganes', 'Malaga', 'Mérida', 'Osasuna', 'Real Madrid',
     'Real Sociedad', 'Sevilla', 'Sporting Gijon', 'Valladolid', 'Valencia', 'Villarreal']
    new_name, score = process.extractOne(teamname, correct_teamnames)
    if score > 80:
        return new_name, score
    else:
        return teamname, score

def plot_year_x_y(df, year, x, y):
    condition1 = (df['Year_start'] == year) &(df['Top4'] == 1) # top 4 teams
    condition2 = (df['Year_start'] == year) &(df['Top4'] == 0) # non top 4 teams
    plt.scatter(df[x][condition1], df[y][condition1], marker ='^', color ='r', s = 100)
    plt.scatter(df[x][condition2], df[y][condition2], marker ='o', color ='b', s = 100)
    plt.title(year, fontsize=16)
    axes = plt.gca()
    axes.set_xlim([0.5, 3.5])
    plt.xlabel('Goals scored', rotation=0, fontsize=16, weight='bold')
    plt.ylabel('Top4', rotation=90, fontsize=16, weight='bold')
    plt.yticks([0,1])
    plt.xticks([1,2,3])
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()

def plot_year_x1_x2(df, year, x1, x2):
    condition1 =(df['Year_start'] == year) &(df['Top4'] == 1)
    condition2 = (df['Year_start'] == year) &(df['Top4'] == 0)
    plt.scatter(df[x1][condition1], df[x2][condition1], marker ='^', color ='r', s = 100)
    plt.scatter(df[x1][condition2], df[x2][condition2], marker ='o', color ='b', s = 100)
    plt.title(year, fontsize=16)
    axes = plt.gca()
    axes.set_xlim([0.5, 3.5])
    axes.set_ylim([0.5, 3.5])
    plt.xlabel('Goals scored', rotation=0, fontsize=16, weight='bold')
    plt.ylabel('Goals lost', rotation=90, fontsize=16, weight='bold')
    plt.xticks([1,2,3])
    plt.yticks([1,2,3])
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()


def models_get_score(models, models_name, X_train, X_test, y_train, y_test):
    '''
    INPUT: models, train/test data
    OUTPUT: dataframe with scores
    '''
    Model_list = []
    Accuracy_list = []
    Precision_list = []
    Recall_list = []
    F1_list = []

    for index, model in enumerate(models):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        Model_list.append(models_name[index])
        Accuracy_list.append(accuracy_score(y_test, y_pred))
        Precision_list.append(precision_score(y_test, y_pred,average = 'macro'))
        Recall_list.append(recall_score(y_test, y_pred,average = 'macro'))
        F1_list.append(f1_score(y_test, y_pred,average = 'macro'))

    score_list = Model_list + Accuracy_list + Precision_list + Recall_list + F1_list
    df = pd.DataFrame(np.array(score_list).reshape(-1,len(Model_list)))
    df = df.transpose()
    df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1']

    return df

def barplot_scores(dataframe):
    ax = dataframe[['Model','Accuracy', 'Precision', 'Recall', 'F1']].plot(x='Model', kind='bar', figsize=(8, 4), legend=False, fontsize=14) # title ="V comp",
    # ax.set_xlabel('Models', fontsize=18)
    ax.set_ylabel('Scores', fontsize=18)
    ax.set_xticklabels(dataframe['Model'], fontsize=18, rotation=0)
    ax.set_ylim([0.5,1])
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.legend(bbox_to_anchor=(1,1), fontsize =20, ncol = 1)

def models_plot_error_curve(models, models_name, X, y):
    plt.figure(figsize=(12,7))

    for index, model in enumerate(models):
        plot_id = '23{}'.format(index+1)
        plt.subplot(plot_id);
        plt.title('{}'.format(models_name[index]), fontsize=20, weight='bold')
        train_sizes, train_scores, valid_scores = learning_curve(model, X, y)
        train_err = 1- train_scores
        ts_err = 1- valid_scores
        train_cv_err = np.mean(train_err, axis=1)
        test_cv_err = np.mean(ts_err, axis=1)
        plt.scatter(train_sizes, train_cv_err, marker= '^', color='forestgreen', s = 100, label = 'Training error')
        plt.scatter(train_sizes, test_cv_err, marker='o', color='orange', s =100, label = 'Testing error')

    plt.xlabel('Train size', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=20, ncol=1)

def knn_plot_k(df, x, y, marker, color, size, label):
    plt.scatter(df[x], df[y], marker =marker, color =color, s = size, label = label)
    axes = plt.gca()
    axes.legend(bbox_to_anchor=(1.5, 1), fontsize =18)
    axes.set_xlim([0, 20])
    plt.xticks([0,5,10,15,20])
    axes.set_ylim([0.7, 1])
    plt.yticks([0.7, 0.8, 0.9,1])
    plt.xlabel('K values', rotation=0, fontsize=16, weight='bold')
    plt.ylabel('Scores', rotation=90, fontsize=16, weight='bold')
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          # title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)

def plot_confusion_matrices(models, models_name, X_test, y_test):
    # plt.figure(figsize=(12,8))
    for index, model in enumerate(models):
        plot_id = '23{}'.format(index+1)
        plt.subplot(plot_id);
        plt.title('{}'.format(models_name[index]), fontsize=16,  weight='bold')
        y_pred = models[index].predict(X_test)
        cm = confusion_matrix(y_test,y_pred)
        # print(cm)
        plot_confusion_matrix(cm,['No','Yes'])


def get_score_model_time(model, ID_model, X, y):
    '''
    INPUT: models, train/test data
    OUTPUT: dataframe with scores
    '''
    X_train_0, X_test_0, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=4444)

    SS = StandardScaler()
    SS.fit(X_train_0)
    X_train = SS.transform(X_train_0)
    X_test = SS.transform(X_test_0)

    ID = []
    Accuracy_list = []
    Precision_list = []
    Recall_list = []
    F1_list = []

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ID.append(int(ID_model))
    Accuracy_list.append(accuracy_score(y_test, y_pred))
    Precision_list.append(precision_score(y_test, y_pred,average = 'macro'))
    Recall_list.append(recall_score(y_test, y_pred,average = 'macro'))
    F1_list.append(f1_score(y_test, y_pred,average = 'macro'))

    score_list = ID + Accuracy_list + Precision_list + Recall_list + F1_list
    df = pd.DataFrame(np.array(score_list).reshape(-1,len(ID)))
    df = df.transpose()
    df.columns = ['Year', 'Accuracy', 'Precision', 'Recall', 'F1']

    return df

def plot_goal_scored_year(df, x, y, size, marker, color, label):
    plt.scatter(df[x], df[y], s=size, marker=marker, color=color, label=label)
    plt.xlabel(x, rotation=0, fontsize=16, weight='bold')
    plt.ylabel(y, rotation=90, fontsize=16, weight='bold')
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.xlabel('Year')
    plt.ylabel('Goals scored')
    axes = plt.gca()
#    axes.set_ylim([1.35,2.6])
#    plt.yticks([1.5,2, 2.5])
    axes.set_xlim([1993,2017])
    axes.set_ylim([0.5, 3.5])
    plt.yticks([1, 2, 3])
    plt.xticks([1995, 2000, 2005, 2010, 2015])
    plt.legend(bbox_to_anchor=(1, 0.8), loc=2, fontsize=16)

def plot_year_player_team(df, player, team, color_player, color_team):
    p1 = plt.bar(df['Year_start'], df[player], color = color_player)
    p2 = plt.bar(df['Year_start'], df[team], bottom = df[player], color = color_team)
    plt.xlabel('Year', fontsize = 18)
    plt.ylabel('Goals scored', fontsize = 18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks([1995, 2000, 2005, 2010, 2015])
    plt.yticks([0,1,2,3])
    # plt.title(player + '/' + team)
    plt.legend((player, team), bbox_to_anchor=(0.1,1.2), loc=2, fontsize=16,ncol=2)
