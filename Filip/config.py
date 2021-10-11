class confuguration:
    # Number of points the of signal left and right from the pick, that is going to be used
    kernel_size = 128 #40

    # Path to the csv dateset in  
    db_path = './MITBIH/'
    
    # List of all classes that are going to be generated
    classes = ['N', 'L', 'R', 'A', 'V']#, '/']# N S V Q 9 4   - P Q R S T

    # Undersample N class
    limit_class_N = 5000000 # 8000 




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
