# Programming: Assignment 7
# Sudoku Solver using Speech Recognition for the Recognition of Spoken Digits

## By Marjolein Spijkerman and Sarah Faste

## The Process: 
### The original Github project: Spoken-Digit-Recognition  
The original project was code to train a HMM model for the recognition of spoken digits based on a small speech dataset based on the original MNST dataset. Their code used Mel-frequency Cepstrum as a way to get the features from the audio dataset. They claimed that their model had an accuracy of 94%. 

### What we planned on doing
Our plan was to use the HMM model created in the original code for the recognition of spoken digits in a sudoku solver. For this, the main steps that we tried to get to work were:
- Finding a way to save the model, so that it could be reused in a different setting without having to retrain the model
- Finding a way to record audio when clicking a button in a GUI
- Finding a way to store the recorded audio and feed it to the HMM model, so that it could predict the correct digit
- Finding a way to combine these predictions with a sudoku puzzle

### What ended up working (and what did not work)
In the end it turned out that training the model again and saving it in a reusable manner was the easiest step. Recording the audio turned out the be a bit trickier, the original plan was to have two buttons; start recording and stop recording. This did not end up working the way we wanted, as we were not able to correctly combine this with the GUI in which we wanted to build our sudoku solver. So, we ended up adding just a record button, that would always record exactly 2 seconds. Which we then combined with a cut silence function, that would remove any white noise around the actual audio that we needed to analyze. Changing this recorded audio in the correct format for the features and using the model to make a prediction based on this feature also did not end up being very problematic. The main problem ended up being that the HMM model simply did not work that well at all. When testing it on the training data, it worked really well. But it did not work well at all for any new data. We tried creating some of our own data to add to the training data set, this seemed to help a little. But, it still did not solve the problem that the code would most likely not work well for most users. So, we decided to change our approach a little. Instead of just inserting the predicted value into the sudoku, we used the fact that for each digit we had a seperate HMM model. Thus, we decided to sort the digits based on their score for each HMM. Sorting them from most likely to least likely. We would then print this list in the GUI and ask the user, to pick the correct digit from the list of possible digits. We kept track of each predicted digit and the corresponding true value, such that we would be able to plot a confusion matrix showing the differences between the actual digits and their predictions. 

## The final version of the code

### Requirements and run instructions 
#### Run Instructions:
Please note that this code can NOT be run in google colab, as both the tkinter and the microphone are not compatible with the colab environment.
To run this code, you can either run it on Windows using Anaconda or in the terminal in Linux. To run the code, simply go the the correct folder and run the following:
- Linux: python3 sudoku_solver.py
- Anaconda: python sudoku_solver.py

![sudoku.png](https://github.com/MarjoleinSpijkerman/Spoken-Digit-Recognition/blob/master/sudoku.png)
This will then open the interface (see the picture above), within the interface there are a couple of things you can do
- Click on "How to use the sudoku solver", to get information on the usage of the interface
- To insert numbers into the sudoku:
  - Click on a square to select that square
  - The selected square will turn light blue
  - To insert a number in the square, either:
    - Click on "record audio" to record 2 seconds of audio
      - It will return a list of possible digits, sorted from most likely to least likely
      - Pick the number you actually said from the menu and click "ok"
      - This number will then be added to the sudoku 
    - Type in a number and click on "manual override" to insert a number manually
      - You have to insert an integer number between 1 and 9. 
      - If you insert anything else, the code give an error message saying you can't insert this
- To empty the selected square, click on the "Reset current button" button
- To empty the entire sudoku, click on the "Reset sudoku" button
- To solve the sudoku click on the "Solve sudoku" button
  - It will solve the sudoku based on the numbers that are currently inserted in the sudoku. 
  - If there are multiple possible solutions, it will display the first found solution
  - If you have inserted any invalid numbers (e.g., the same digit twice in the same row), the code will give an error message and reset all squares. 
- To look at the confusion matrix click on the "Statistics" button
  - It will show a confusion matrix based on the recorded audio
  - It shows the predicted digits vs. the true digits
  - It will not work if the solver has not done any predictions yet

#### Running the code on Anaconda and necessary packages: 
To get the code to work on anaconda do the following:
- install Anaconda
- create a new environment: conda create -n env python=3.7
- activate the environment: conda activate env

Then install all the necessary packages:
- conda install -c conda-forge python-sounddevice
- conda install -c anaconda scipy
- conda install -c conda-forge librosa
- conda install seaborn
- conda install -c conda-forge hmmlearn

### Folder structure 

This github page has 4 main components
- "new-spoken-digit", which is a folder containing the dataset
- "SDR.py", which is used to train an HMM model on the dataset
- "model_hmm.pkl", a pickle file, which contains the trained HMM model
- "sudoku_solver.py", which contains a sudoku solver which uses the HMM model for speech recognition

The SDR file is almost identical to the original version. The main difference is that the current version saves a pickled version of the model.
The dataset contains the entire original dataset, but with 20 added audio files for each digit recorded by us (10 each). 
The sudoku_solver code is completely new. 

