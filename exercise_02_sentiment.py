from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.naive_bayes import MultinomialNB

if __name__ == "__main__":

    movie_reviews_data_folder = r"./data"
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    #print("n_samples: %d" % len(dataset.data))
    
    docs_new = ['Amazing', 'loved', 'Good']

    # Alterando o text_size para 0,40 a precisão de ambos métodos de classifcação melhoraram
    docs_train, docs_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.40, random_state=None)
    
    #Classificador Support Vector Machine
    text_clf_v = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),
                ])
    text_clf_v.fit(dataset.data, dataset.target)  

    predicted = text_clf_v.predict(docs_new)
    predicted = text_clf_v.predict(docs_test)
    print('Vector Machine Precisão')
    print(np.mean(predicted == y_test))
    #
    
    #Classificador Naive bayes
    text_clf_b = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])

    text_clf_b.fit(dataset.data, dataset.target)  
    predicted = text_clf_b.predict(docs_new)
    predicted = text_clf_b.predict(docs_test)
    print('Naive bayes Precisão')
    print(np.mean(predicted == y_test))

    #GridSearchCV
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              }
    gs_clf = GridSearchCV(text_clf_v, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(dataset.data, dataset.target)
    dataset.target_names[gs_clf.predict(['This film is amazing!'])[0]]
    print('GridSearchCV Com Vector Machine:')
    print(gs_clf.best_score_)
    #for param_name in sorted(parameters.keys()):
        #print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        
    gs_clf = GridSearchCV(text_clf_b, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(dataset.data, dataset.target)
    dataset.target_names[gs_clf.predict(['This film is amazing!'])[0]]
    print('GridSearchCV Com Naive bayes:')
    print(gs_clf.best_score_)
    print(gs_clf) 
    #for param_name in sorted(parameters.keys()):
        #print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

#Nos testes acima implementei dois classficadores o Support Vector Machine e o Naive Bayes e durante os testes percebi que era possível aumentar a precisão do algoritmo
#Se eu aumentasse o text_size dos seus default 0.25 para 0.40, ambos os métodos na media aumentaram bastante a sua precisão,
#mas o maior destaque fica Naive bayes que em todos os meus testes teve um valor superior ao vector machine.
#Já ao aplicar o GridSerach o resultado foi o inverso, onde o vector machine conseguiu adquirir uma pequena vantagem sobre o naive bayes.
#Os prints dos best params estão comentados com o objetivo de despoluir o console de prints 
