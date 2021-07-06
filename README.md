# Elliot_refactor-tesi-Ventrella
Framework Elliot con modifiche effettuate durante il lavoro di tesi di Ventrella

## Dataset
Il percorso dei due dataset utilizzati negli esperimenti è ```data/ventrella_experiment/```:
  - dataset movielens -> ```train2id_movielens.csv``` + ```test2id_movielens.csv```
  - dataset dbbook    -> ```train2id_dbbook.csv``` + ```test2id_dbbook.csv```

## Modifiche su Elliot
Sono state effettuate le seguenti modifiche al framework:
  1. è stata implementato l'```F1-score macro-averaged``` a fronte del solo micro-averaged presente di default (inserire fra le ```simple_metrics``` '```MavF1```')
  2. la ```Precision``` è stata modificata per non ignorare nel calcolo utenti che nel test set non hanno item rilevanti associati
  3. è stato creato il modello ```Bypass``` in ```external/bypass/```, che permette di saltare l'addestramento di un modello e passare direttamente alla valutazione, utilizzabile richiamandolo nel file di configurazione (inserire il file dei risultati in ```train_path``` ed il test set in ```test_path```)
  4. inserito il metodo ```test_item_only_filter``` all'interno di ogni modello (metodo ```get_recommendations()```) per l'applicazione della strategia ```TestRatings``` (considera per la valutazione solamente gli item presenti in ground truth)
  5. creati diversi modelli esterni (ognuno associato ad un file di risultati in ```top-5```) per permettere la valutazione simultanea delle baseline e dei modelli ibridi (```external/models/```)

## Side information
Sono state create le side information per ognuno dei dataset, in formato leggibile da Elliot (```data/ventrella_experiment/side_informations/```)

## Note
  - utilizzare un valore di ```top_k``` grande quanto il dataset quando si utilizza la strategia ```TestRatings```
  - con il modello ```Bypass``` è possibile la valutazione di un solo file di risultati alla volta

## File di esempio
Sono disponibili 3 esperimenti pre-impostati:
  - ```sample_bypass.py``` associato al file di configurazione ```configuration_bypass.yml```, che utilizza il bypass del training per effettuare le sola valutazione
  - ```sample_side_informations.py``` associato al file di configurazione ```side_information_configuration.yml```, dove viene effettuato l'addestramento dei modelli utilizzando le side information
  - ```sample_baselines.py``` associato al file di configurazione ```baseline_configuration.yml```, che fa partire la valutazione (a seconda del dataset indicato in ```train_path``` e ```test_path```) dei modelli di baseline e dei modelli ibridi (TransE/TransH e BERT/USE)
