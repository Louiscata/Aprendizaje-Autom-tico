# Competición de Kaggle

O obxectivo da práctica é evaluar o desempeño de diveros modelos de comités, principalmente bagging, boosting, stacking e blending. 

Ademais, deberanse poñer en uso coñecementos das outras prácticas, como o preprocesado de datos, normalización, análise e comparación entre modelos, etc.

## Observaciones sobre o dataframe

Ten un total de 30 columnas, sendo unha delas 'Target_Risco', o label que buscamos predicir. Este é un valor de 0 a 3 que indica menor ou maior risco.

Outra columna é o ID de cliente, que ignoraremos.

Algunhas variables son categóricas e cómpre empregar One-Hot encoding. Para outras que son continuas pero teñen moitos datos próximos ao 0 (ten sentido que haxa máis xente con menos patrimonio) tentaremos empregar un logaritmo.

Todas as columnas teñen valores válidos excepto patrimonio neto e tempo en encher o formulario, con uns poucos NaN.

## Estructura de carpetas
### ./scripts
Os "mains" que vamos executando según cambiamos cousas
### ./resultados
Donde guardamos os csv de saída (predicións de test) para subir a Kaggle
### ./scripts/funciones
Submódulos con funcións auxiliares para compartimentar o main

## Estratexias intentadas

Probamos inicialmente a meter os datos sen máis nun clasificador árbore ou MLP, obtendo un score de 0.71

Pronto fixemos un ensemble tipo stacking con 4 modelos (Random forest, gradient boosting, MLP, KNN), chegando a 0.78

Todo o anterior foi sen preprocesado prácticamente. Cando empezamos a probar a preprocesar de diferentes xeitos, os modelos deixaban bastante que desear.

Tiñamos algunhas fallas como que, por exemplo, se quitamos un outlier e o imputamos pola mediana, estamos volvendo "pobre" a alguén con millóns de euros.

Tampouco daba demasiado resultado converter algunhas variables en logaritmos ou facer "clamp" para poñer os outlier como o cuantil 99%

Regresamos a algo moi sinxelo sen preprocesado empregando un random forest, LGBM e XGB que acadou puntuación de 0.79