J'ai divisé le run en 3 fichier. 

- Le premier \textbf{build data.py} prend le fichier \textbf{data/sequoia-corpus+fct.mrg\_strict} et sort les fichiers \textbf{training set} (annoté), \textbf{validation set} (annoté) et \textbf{raw validation set} (non annoté).

 - Le second \textbf{train.py} prend 2 ou 3 argument, il s'appelle comme suit : \textit{python train.py (ANNOTATED\_TRAINING\_SET) (PREFIX OF GENERATED MODEL) (facultative - EMBEDDED LEXICON)}. Il prend en entrée le fichier training set et sort dans le dossier \textit{model} 4 fichier pour sauvegarder le PCFG (lexicon et grammar), le lexique vectorisé et le fichier Polygot recopié. Le lexique vectorisé peut éventuellement être remis en argumant pour éviter de le régénérer pour gagner du temps. Il met environ 20 min à être généré. Il est proposé de le mettre dans le rendu du TP2 pour éviter d'avoir à le régénérer. 

- Enfin le dernier fichier \textbf{test} s'appelle comme suit :\textit{python test.py (PREFIX OF THE MODEL) (TESTING SET NOT ANNOTATED) (TESTING SET ANNOTATED)}. 
 
 Enfin \textbf{run.sh} réalise la pipeline de bout en bout sans demander d'arguments.