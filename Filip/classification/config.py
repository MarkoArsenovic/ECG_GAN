class confuguration:
    kernel_size = 80 #40
    db_path = './../DateSet/ECG/csv/'
    classes = ['N', 'L', 'R', 'A', 'V', '/']
    limit_class_N = 50000 # 8000 




"""

Prebrojati klase

npr.

N 100
A 10 

// Obrnuto proporcijalno

Isprobati za razlicito a i b 

class_weight: {N = 10 * a, A = 100 * b}

Za razlicita klasa 'kernel'

Grid search - ceo ds svaki put
Parametar za score - scoring="balanced_accuracy" 

Halving - ide od pola, pa pola eleminise, pa pola ucita
Parametar za score - scoring="balanced_accuracy" 





"""