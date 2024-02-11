# IMPORTANDO BIBLIOTECAS
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import random
from scipy import stats

# IMPORTANDO DADOS
dados1 = np.load('Dados_1.npy')
dados2 = np.load('Dados_2.npy')
dados3 = np.load('Dados_3.npy')
dados4 = np.load('Dados_4.npy')
dados5 = np.load('Dados_5.npy')
dados5[np.isnan(dados5)]=0
classe = np.load('Classes.npy',allow_pickle=True)

# DELETAR COLUNAS NAN
dados1=np.delete(dados1,[200],axis=1)
dados2=np.delete(dados2,[200],axis=1)
dados3=np.delete(dados3,[200],axis=1)

# MESCLAR DADOS
dados_mesclados = np.concatenate((dados1,dados2,dados3),axis=1)

# FUNÇÃO PARA REMOÇÃO DE OUTLIERS
def outliers(data):
    mean = np.mean(data)
    std = np.std(data)
    z_score=data-mean/std
    outliers=abs(z_score)<=3
    outliers=1*outliers
    data=np.multiply(data,outliers)
    return data

# REMOVENDO OUTLIERS
dados1=outliers(dados1)
dados2=outliers(dados2)
dados3=outliers(dados3)
dados_mesclados=outliers(dados_mesclados)

# TRANSFORMAR CLASSE OBJ EM FLOAT
classe=classe.astype(str)
output = np.empty([50000, 1], dtype=float)
output[classe=='Classe A']=0
output[classe=='Classe B']=1
output[classe=='Classe C']=2
output[classe=='Classe D']=3
output[classe=='Classe E']=4

#DEFINIR FUNÇÃO PARA TREINAMENTO DOS DAS REDES COM OS DADOS
def treinamento(data):
    # SEPARANDO CONJUNTOS DE TREINO, VALIDACAO E TESTE
    global Y_test
    X_train, X_out, Y_train, Y_out = train_test_split(data, output, test_size=0.4,random_state=0)
    X_val, X_test, Y_val, Y_test = train_test_split(X_out, Y_out, test_size=0.5,random_state=0)
    
    # CRIANDO MODELO
    inputs=len(data[0])
    model = keras.Sequential([
        keras.layers.Dense(100,input_shape=(inputs,), activation='relu'),  #flateen to reshape an image
        keras.layers.Dense(20,activation='relu'),
        keras.layers.Dense(5,activation='softmax')
        ])
    
    
    model.summary() #verificando o modelo
    
    # PARAMETROS DE TREINAMENTO
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # TREINANDO O MODELO
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3)
    mc = ModelCheckpoint('best_model_test.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    history=model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=30,callbacks=[es,mc])
    
    # GRÁFICO DO PROGRESSO DE TREINAMENTO
    history.history.keys()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisão do modelo')
    plt.ylabel('Precisão')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='lower right')
    plt.show()
    
    # GRÁFICO DO PROGRESSO DE TREINAMENTO
    history.history.keys()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Função custo do modelo')
    plt.ylabel('Custo')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='lower right')
    plt.show()
    
    # AVALIANDO O MODELO
    model.load_weights('best_model_test.h5') #carrega o modelo que teve melhor desempenho de acerto para a validação no treinamento
    test_loss,test_acc=model.evaluate(X_test,Y_test)
    print("loss=%.3f" %test_loss)
    print("accuracy=%.3f" %test_acc)
    
    # PREDIZER A CLASSE DE UM EXEMPLO DO CONJUNTO DE TESTE ALEATÓRIO
    n=random.randint(0,9999)
    predictes_value_dadosmesclados=model.predict(X_test)
    print("A classe é = %d" %np.argmax(predictes_value_dadosmesclados[n]))
    
    # TRANFORMAR RESULTADO CATEGÓRICO PARA VALOR ÚNICO
    resultados_classe = np.zeros((10000,1))
    for n in range(10000):
        resultados_classe[n] = np.argmax(predictes_value_dadosmesclados[n])
    return resultados_classe

# APLICAR FUNÇÃO DE TREINAMENTO PARA OS DADOS
resultados_dados_mesclados=treinamento(dados_mesclados)
resultados_dados1=treinamento(dados1)
resultados_dados2=treinamento(dados2)
resultados_dados3=treinamento(dados3)
# CONCATENAR DADOS PARA CRIAR RESULTADO FINAL MAIS ROBUSTO (DADOS MESCLADOS ADICIONADOS 2X POIS TEM MELHOR RESULTADO, ADICIONANDO MAIS PESO)
resultados = np.concatenate((resultados_dados1,resultados_dados2,resultados_dados3,resultados_dados_mesclados,resultados_dados_mesclados),axis=1)
# CALCULAR MODA DOS RESULTADOS E COMPARAR COM OS TARGETS
mode,count=stats.mode(resultados,axis=1)
mode=np.reshape(mode,(10000,1))
final=mode==Y_test
trues = np.count_nonzero(final)
resultado_final=trues/100
print("Porcentagem de acerto do modelo foi de %d" %resultado_final)