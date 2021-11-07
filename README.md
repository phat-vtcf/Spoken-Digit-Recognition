This github has 4 main components
- a folder containing the dataset
- SDR.py, which is used to train an HMM model on the dataset
- a pickled model, which contains the trained model
- sudoku_solver.py, which contains a sudoku solver which used the HMM model for speech recognition

The SDR file is almost identical to the original, with the main difference being that it now saves a pickled version of the model
The main importance is the sudoku_solver code which:
- creates a GUI in ktinker where you can
- insert your sudoku either using voice or text based commands
- solve the sudoku
- reset the sudoku
- see a confusion matrix comparing the guesses of the model with the actual input
- it also contains an information page, with more information regarding the program

Please note that this code can NOT be run in google colab, as both the ktinker and the microphone are not compatible with the colab environment.
To run this code, you can either run it on linux by calling 
- python3 sudoku_solver.py
or on windows in the anaconda environment
