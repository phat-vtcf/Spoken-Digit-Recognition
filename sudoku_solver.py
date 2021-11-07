#All the necessary imports 
from tkinter import * 
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import librosa
from librosa.feature import mfcc
from sklearn.metrics import confusion_matrix
import pickle
import tkinter.font as font
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GMMHMM

#setup to use matplotlib inside the tkinter environment
matplotlib.use('TkAgg')

#create the main tkniter environment, plus add title and size 
root = Tk()
root.title('Sudoku')
root.state('zoomed')

#Defining the sudoku (including the ability to solve itself) as a class 
class Sudoku:
	#intialize the sudoku as an empty 9x9 grid filled with 0s
	def __init__(self):
		self.sudoku = [[0,0,0,0,0,0,0,0,0],
					   [0,0,0,0,0,0,0,0,0],
					   [0,0,0,0,0,0,0,0,0],
					   [0,0,0,0,0,0,0,0,0],
					   [0,0,0,0,0,0,0,0,0],
					   [0,0,0,0,0,0,0,0,0],
					   [0,0,0,0,0,0,0,0,0],
					   [0,0,0,0,0,0,0,0,0],
					   [0,0,0,0,0,0,0,0,0]]
									   
	
	#update a square of the sudoku with a new value 
	def update_value(self, column, row, value):
		self.sudoku[row][column] = int(value)
	
	#figure out what digits are possible in a square
	def find_possible_digits(self, row, column, digit):
		#first go over the rows and columns, to see if the digit already exists
		for x in range(9):
			if self.sudoku[row][x] == digit:
				return False
			 
		for x in range(9):
			if self.sudoku[x][column] == digit:
				return False
		
		#Now check the 3x3 block, to see if the digit already exists
		start_row = row - row % 3
		start_col = column - column % 3
		for i in range(3):
			for j in range(3):
				if self.sudoku[i + start_row][j + start_col] == digit:
					return False
		
		#digit does not exist yet, can be inserted into sudoku
		return True
	
	#We're solving the sudoku using the standard recursive method
	#Fill in a possible digit in each square
	#When you run out of options, delete previous changes and try the next number
	#Untill a valid solution is found 
	def solve_sudoku_recursion(self, row, column):	
		#Since we go over row 0-8, once we reach row==9 we have finished the sudoku
		if row == 9:
			return True
		
		#if we're not at the last column yet, the next option we need to look at is row, column+1
		if column < 8:
			r = row
			c = column+1

		#If we're at the last column, the next option we need to look at is row+1, column=0
		else:
			r = row+1
			c = 0
		
		#This one is already filled, continue with the next value
		if self.sudoku[row][column] != 0:
			return self.solve_sudoku_recursion(r, c)
		
		#Go over digit 1-9
		for digit in range(1,10):
			#check if the digit is possible
			if self.find_possible_digits(row, column, digit):
				
				#insert the found digit, and continue with next empty square
				self.sudoku[row][column] = digit
				if self.solve_sudoku_recursion(r, c):
					return True
			#If there was a problem, set the value back to 0
			self.sudoku[row][column] = 0			
		return False
	
	#returns the value in a specific square 
	def return_value(self, row, column):
		return self.sudoku[row][column]			

#This class keeps track of which of the 9 sudoku squares is currently selected
#Default selected button is position 0,0
class Current():
	def __init__(self):
		self.current_button = (0,0)
		self.current_button_col = 0
		self.current_button_row = 0
	def update(self, i, j):
		self.current_button = (i, j)
		self.current_button_col = i
		self.current_button_row = j
		
#initialize the currently selected button object
current_button = Current()
#initialize the sudoku object
sudoku = Sudoku()

#create the lines around the sudoku buttons 
canvas = Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight(), 
				   borderwidth=0, highlightthickness=0)
#thick lines horizontal
canvas.create_line((75, 75), (795, 75), width=4)
canvas.create_line((75, 315), (795, 315), width=4)
canvas.create_line((75, 555), (795, 555), width=4)
canvas.create_line((75, 795), (795, 795), width=4)
#thin lines horizontal
canvas.create_line((75, 155), (795, 155), width=1)
canvas.create_line((75, 235), (795, 235), width=1)
canvas.create_line((75, 395), (795, 395), width=1)
canvas.create_line((75, 475), (795, 475), width=1)
canvas.create_line((75, 635), (795, 635), width=1)
canvas.create_line((75, 715), (795, 715), width=1)

#thick lines vertical
canvas.create_line((75, 75), (75, 795), width=4)
canvas.create_line((315, 75), (315, 795), width=4)
canvas.create_line((555, 75), (555, 795), width=4)
canvas.create_line((795, 75), (795, 795), width=4)
#thin lines vertical
canvas.create_line((153, 75), (153, 795), width=1)
canvas.create_line((233, 75), (233, 795), width=1)
canvas.create_line((393, 75), (393, 795), width=1)
canvas.create_line((473, 75), (473, 795), width=1)
canvas.create_line((633, 75), (633, 795), width=1)
canvas.create_line((713, 75), (713, 795), width=1)

#insert the lines into the GUI
canvas.pack()

#This creates the buttons that make up the main part of the GUI
#Each button comes with the onClick function, that updates the current button
#The color of the currently pressed button gets changed to light blue to keep easier track of which button was last clicked
global buttons
buttons = []

def onClick(i):
	#default color
	Default = 'SystemButtonFace'
	#We now have a different current button, so the previous one should go back to default color
	buttons[current_button.current_button_col][current_button.current_button_row]['bg'] = Default
	
	#update the current button to the newly clicked button 
	current_button.update(int(i[0])-1, int(i[1])-1)
	buttons[current_button.current_button_col][current_button.current_button_row]['bg'] = '#bcedef'

#we create 9 x 9 buttons for the sudoku grid 	
for i in range(1, 10):
	#define the used font 
	myFont = font.Font(size=9, weight="bold")
	buttons_column = []
	for j in range(1, 10):
		str_pos = str(i) + str(j)
		b = Button(root, height=4, width=8, text = "", command = lambda str_pos=str_pos: onClick(str_pos))
		b['font'] = myFont
		#place them evenly
		b.place(x=i*80, y=j*80)
		buttons_column.append(b)
	buttons.append(buttons_column)


#Open the saved HMM model, that we'll use to predict the spoken digits
with open('model_hmm.pkl', 'rb') as f:
	hmmModels = pickle.load(f)


#Cut the silence from the recordings (slight variation on the program created for the regular assignments)
def cut_silence(data, sr, threshold = 0.1 , padding = 400):
	min_val = threshold * max(abs(data))
	start_audio_pos = 0
	end_audio_pos = 0

	for i in range(len(data)):
		if data[i] > min_val:
			start_audio_pos = i
			break

	for i in range(len(data)-1, 0, -1):
		if data[i] > min_val:
			end_audio_pos = i+1
			break
  
	data = data[start_audio_pos : end_audio_pos]
	value = int(sr * padding * 0.001)
	zero_padding = np.zeros(value, dtype=int)
	final_audio = np.concatenate((zero_padding, data, zero_padding))
	return final_audio

#These two functions work as follows:
#- it starts in record audio; 
#- it will record 2 seconds of audio, when the button is clicked
#- this then gets saved into wav format, so that we can use librosa load to get it in the wanted format
#- we use libroasa's mfcc features function to get the Transpose of the features, which the original hmm model was trained on
#- we then go over the HMM models of each digit and take the score for each digit
#- we sort the different scores from highest to lowest 
#- create a list with all the possible options, the option that is picked is the option that will be added to the sudoku
#- we also keep track of the prediction and the true value, so that we can run a confusion matrix later on
def submit_solution(variable1):
	#grabs the correct solution from the option menu
	solution = int(variable1.get())
	#add the true solution to the list of solutions (will be used to create confusion matrix)
	solutions_prediction.update_true(solution)
	#update the value in the sudoku and in the GUI
	buttons[current_button.current_button_col][current_button.current_button_row]['text'] = solution
	sudoku.update_value(current_button.current_button_col, current_button.current_button_row, solution)
	#destroy the pop up screen 
	toplevel1.destroy()
	
def record_audio():
	fs = 44100	# Sample rate
	seconds = 2	 # Duration of recording

	my_recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
	sd.wait()  # Wait until recording is finished
	write('output.wav', fs, my_recording)  # Save as WAV file 
	
	wave, sample_rate =	 librosa.load('output.wav') #load into librosa format
	wave = cut_silence(wave, sample_rate) #cut the silence
	mfcc_features = mfcc(wave, sample_rate).T #get the transpose of the features
	
	scoreList = {}
	for model_label in hmmModels.keys():
		model = hmmModels[model_label]
		score = model.score(mfcc_features)
		scoreList[model_label] = score #get score for each possible digit
	
	dic2=dict(sorted(scoreList.items(),key= lambda x:x[-1], reverse=True)) #sort the digits by score
	
	text = "The prediction of the spoken digit, sorted by likelihood is: \n\n" 
	options = []
	for item in dic2:
		text = f"{text}{item}\n" #create text message for pop up screen
		options.append(item) #create the list of pickable options, sorted by likelihood
	
	global toplevel1 
	toplevel1 = Toplevel() #create a new pop up screen 
	label2 = Label(toplevel1, text=text, height=0, width=1000) #add the created text message to the pop up screen 
	label2.pack()
	
	variable1 = StringVar(toplevel1) #create the menu variable 
	variable1.set(options[0]) #The most likely digit is the default value of the menu
	w = OptionMenu(toplevel1, variable1, *options) #create the option menu
	w.pack() #add the menu to the pop up screen 
	correct_button = Button(toplevel1, text = 'Correct Solution', command = lambda variable1 = variable1: submit_solution(variable1)).pack() #create the button that is used in the pop up screen 
	solutions_prediction.update_pred(int(options[0])) #add the predicted solution to the list of solutions (will be used to create confusion matrix)

#create the record audio button 
myFont = font.Font(size=18, weight="bold")
button_rec = Button(root, text='Record audio', font = myFont, command=record_audio)
button_rec.place(x=900, y=200)

#create the solve sudoku button 
#clicking it will solve the sudoku and update the GUI
def solve_sudoku():	
	sudoku.solve_sudoku_recursion(0, 0)
	
	for i in range(9):
		for j in range(9):
			buttons[i][j]['text'] = sudoku.return_value(j, i)
			
myFont = font.Font(size=18, weight="bold")	
button_solve = Button(root, text='Solve sudoku', font=myFont, command=solve_sudoku)
button_solve.place(x=900, y=540)



#Create the manual insertion of numbers into the sudoku
#This function takes a text message and insert it into the sudoku after clicking the button
#- first click a square you want to update, type a number, click the button, number is now in sudoku
value = StringVar()
def submit():
	number = value.get()
	if number.isdigit():
		number = int(number)
		if number > 0 and number < 10:
			buttons[current_button.current_button_col][current_button.current_button_row]['text'] = number
			sudoku.update_value(current_button.current_button_col, current_button.current_button_row, number)

myFont = font.Font(size=18, weight="bold")
value_entry = Entry(root,textvariable = value, font = myFont).place(x=880, y=260)
sub_btn=Button(root,text = 'Manual override', font = myFont, command = submit).place(x=900, y=300)

#the reset buttons, either will reset the current square or reset the entire sudoku
def reset_one():
	buttons[current_button.current_button_col][current_button.current_button_row]['text'] = ''
	sudoku.update_value(current_button.current_button_col, current_button.current_button_row, 0)
	

def reset_all():
	for i in range(9):
		for j in range(9):
			buttons[i][j]['text'] = ''
			sudoku.update_value(i, j, 0)

myFont = font.Font(size=18, weight="bold")	
button_reset_1 = Button(root, text='Reset current button', font=myFont, command=reset_one)
button_reset_1.place(x=900, y=420)
	
button_reset_all = Button(root, text='Reset sudoku', font=myFont, command=reset_all)
button_reset_all.place(x=900, y=480)



#Creation of the about page, which creates a pop-up screen containing some information regarding the program 
Information_Text = """How to use the sudoku solver:

First select a square in the sudoku --> the currently selected square will turn blue
Then either select record audio to insert a number between 1 and 9 by means of speaking or manually insert a number
When picking the record audio options, the program will do the following
  - record for 2 seconds
  - create a new pop-up window containing:
    -- a list with all digit predictions, sorted on most likely to least likely
    -- a menu, where you need to pick the digit you just said out loud

Click on reset current button, to empty the current square
Click on reset all to empty the entire sudoku

Click on solve sudoku to get a solution
Please note:
  -	 If there are multiple solutions, the solver will return only 1 solution
  -	 If there are errors in the given input, the solver will not be able to find a good solution

After using the record option at least once, the statistics button will show a confusion matrix.
Please note that it won't show a confusion matrix, if it has no data yet. 
  """

def clickAbout():
	toplevel = Toplevel()
	label1 = Label(toplevel, text=Information_Text, height=0, width=100)
	label1.pack()


button1 = Button(root, text="How to use the sudoku solver", font=myFont, command=clickAbout)
button1.place(x=900, y=100)




##########PLOT A CONFUSION MATRIX
#This keeps track of all spoken digits predictions and true solutions
#will be used for the confusion matrix 
class Confusion_Matrix():
	def __init__(self):
		self.true = []
		self.prediction = []
	
	def update_true(self, t):
		self.true.append(t)
		
	def update_pred(self, p):
		self.prediction.append(p)
	
	def return_true(self):
		return self.true
		
	def return_pred(self):
		return self.prediction
		

def create_matrix():
	#create the pop-up screen 
	toplevel = Toplevel()
	
	#get a list of predictions and solutions 
	y_pred = solutions_prediction.return_pred()
	y_true = solutions_prediction.return_true()
	
	#make the confusion matrix 
	cm = confusion_matrix(y_true, y_pred)
	
	#create the labels (only the ones that were used so far)
	labels = []
	for item in y_true:
		labels.append(str(item))
	class_names = labels
	
	#create the plot 
	fig = plt.figure(figsize=(16, 14))
	ax= plt.subplot()
	sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
	ax.set_xlabel('Predicted', fontsize=20)
	ax.xaxis.set_label_position('bottom')
	plt.xticks(rotation=90)
	ax.xaxis.set_ticklabels(class_names, fontsize = 10)
	ax.xaxis.tick_bottom()

	ax.set_ylabel('True', fontsize=20)
	ax.yaxis.set_ticklabels(class_names, fontsize = 10)
	plt.yticks(rotation=0)
	
	canvas = FigureCanvasTkAgg(fig, master=toplevel)
	canvas.get_tk_widget().pack()
	canvas.draw()
	
#button that will call upon a new screen containing a confusion matrix 		
solutions_prediction = Confusion_Matrix()
stats_button = Button(root, text="Statistics", font=myFont, command=create_matrix)
stats_button.place(x=900, y=700)


root.mainloop()