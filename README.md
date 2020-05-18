# pylearn
LearningWithPython

This is a fun **first** python learning repository



Udacity ND: Intro ML with TENSORFLOW

Lessons and quizzes support learning and are hence optional
Project completion is compulsory component to the ND.

Three components to the course:
1: Supervised
SKLearn training
Algorithm theory
Combining methods
Topics
Linear
Logistic
DTs
NaiveB
NNs
SVMs
2: Deep
NNs for patterns
Numpy: Backpropagation and optimisation
Tensorflow Image classifier
3: Unsupervised
Clustering dimension reduction

Extracurricular
Prerequisites	= Supporting core material (Git/Hub, SQL, Python data, CommandLine)
Additional 		= Optional for deeper / applied understandings (LAlgebra, Stats, Visualisation)

Ned Ideas
GitHub ‘pages’ ~ include index.html file in repo to instantly show off your work
It’s perfect light webpage: no hosting , no FTP and no DNS... wow!

Anaconda Navigator = “Desktop Portal to Data Science”
Anaconda = MiniCondat (= Conda+ Python) + lots of pre-compiled packages
It’s large (~500Mb) due the number of packages it contains
Its packages tend to slower to be released but more stable than those directly available from developers

Conda is a package manager, similar to Pip.
Pip handles python packages
Conda handles any type of package
Ready to install 1,500 packages (including R ones) .. and there’s more packages it can install! (Other sources)
Use with: conda install PACKAGENAME

MKL - (Intel) Math Kernel Library
Set of threaded and vectorised routines to accelerate math functions/applications.
Optimisations included in Anaconda are: Numpy and SciKit-Learn

Cholesky decomposition
Neat trick to efficiently solve solve linear systems of equations Ax=b.
Breaks the problem into smaller, easy/efficient to solve steps
Applications: least squares regression .. plus others
Method:
Matrix A can be written as LL^T where L is lower triangular matrix
Want to find x, for A and b, where A is positive definite
Ax=b
LL^Tx=b
becomes
L^Tx=c
Lc=b
c can be efficiently solved for, and then so can x.
Good source: https://towardsdatascience.com/behind-the-models-cholesky-decomposition-b61ef17a65fb

Setting up (Ana)Conda
Download v3.7 64bit
'conda list’ command should reveal mainly 3.7 versions (not ~2.7)
conda update conda
conda update —all

conda install nb_conda (notebook version of conda)

ipython ‘Kernel'
Cleaner command line kernel compared to running just ‘python’ from command line (run ‘ipython’)

Access markdown language from within jupiter by changing the markdown cell type.
>Code syntax:
````
import numpy as np
*Numpy imported*
````
>Math syntax:
$$
y = \frac{a}{b+c}
$$
> Header/r markdown syntax 
#         this will be the largest header font
##       this will be the second largest header font
##### this will be the fifth largest header font
> R markdown syntax cheat sheet here: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

Code cells.. any code written (like defining a variable) is available in other code cells

Jupiter Keyboard shortcuts depend on main two perspectives
Whether keyboard is in EDIT (Green) or COMMAND (blue) mode

Common Shortcuts
Enter .. Change cell into EDIT mode
Esc   ..  Change cell into COMMAND mode


A   .. Add cell above current cell
B   .. Add cell below current cell
S   .. Save
M .. Merge Cell
D,D (repeated) .. Delete Cell
Shift + Cmd + P  .. Command palette (use this for searching for commands:
if we search for ‘change’, we can change cell to:
Raw
Code
Markdown
Switching to wrong cell type may make it harder to execute that step.
if we search for ‘move’, we are provided option to shift cells up or down.

Leverage Latex formulae for nice formatting, such as y = a / (b+c)
$$
y = \frac{a}{b+c} 
$$


Getting help
After importing numpy, it’s possible to run:
help(numpy.add)   .. i.e. the addition within bumpy
but ordinarily, help(bumpy) returns error:
	IOPub data rate exceeded.
	The notebook server will temporarily stop sending output
	to the client in order to avoid crashing it.
This error can be solved by uncommenting-out the following line (using nano + ctrl-W search):
#c.NotebookApp.iopub_data_rate_limit = 0


Magic Keywords
There are many (>10) of these
Specific to regular Python kernel
%     .. Line magics
%% .. Cell magics
Full list here: https://ipython.readthedocs.io/en/stable/interactive/magics.html

%time, %timeit, %%time
%matplotlib
  >> Ensure jupyter shows plots (need ‘inline’ parameter)
  >> Possibly most critical command for report-based notebook??
%system
  >> enables access to the shell (e.g. '%system pwd’ returns [‘/Users/../.. /folder’])
%conda install [pkgs]
%dirs
  >> current directory
%env HOME    … /Users/Administrator
  >> returns environmental variables
%history
  >> return jupiter’s history
%load ~/hello_world.py
  >> run code from external script
%pdb .. debugging in the notebook (more sophisticated traceback?)

Plotting
  — Simple first working example in python command line AND jupyter.
import matplotlib.pyplot as mplpp
mplpp.plot([4,5,6,2])
mplpp.show()

For interactive variants, need to pass argument to specify ‘backend’ to do the rendering.
By default is the ’server side’ .. Python command line
instead can specify rendering in notebook (~interactive) using magic command ‘%matplotlib inline’.
Example:
  >> cell1
%matplotlib notebook
import matplotlib.pylab as plt
import numpy as np 
  >> cell2:
my_x = np.linspace(0,2*(2*np.pi),100)
my_y = np.sin(my_x)
line, = plt.plot(my_x,my_y,'--',linewidth = 4)

Notebook conversion
Notebooks are examples of JSON files with .ipynb extension(JavaScript Object Notation)
Text format that is language independent.
	Object defined by {}			(curly braces)
	each name followed by : 		(colon)
	name/value pairs separate by , 	(comma)
Thus because notebooks are JSON, utilities can simply convert jupyter notebooks to other formats.
In-built Jupyter tool: nbconvert for converting to html, markdown or slideshow.
Benefit:
	Enables sharing of jupyter notebook (including results) to people without Jupyter
Examples:
	jupyter nbconvert --to html notebook.ipynb
	jupyter nbconvert --to markdown /Users/Administrator/Documents/Work/pylearn/PrintPlay.ipynb 
	jupyter nbconvert --to slides /Users/Administrator/Documents/Work/pylearn/PrintPlayforSlides.ipynb  (see slide notes)
Current formats supported:
 — HTML,
 — LaTeX,
 — PDF,
 — Reveal.js HTML slideshow
 — Markdown
 — Ascii
 — reStructuredText
 — executable script
Help:
	Command line help is available via ‘jupyter nbconvert’ in terminal
Note on slides conversion:
In order for slides (in html format) to work / render / convert, need to specify whether cells are:
	slides (left-right scroll)
	sub-slides (up-down from main slides)
	fragments - pieces of text that appear
Add cell metadata via menu dropdown: View > Cell Toolbar > Slideshow



Data scientist : confluence of statistics and computer science

Common distributions
Number of correct answers to a test: Binomial distribution
Number of random incidents: Poisson distribution
Word frequency in a test: Zipf law
  >> discrete power low distribution, originally formed in quantativie linguistics
  >> It says frequency is of a word, is inversely proportional to its rank.
  >> first most frequent word may be 8% of words, then the second most frequent would be 4% (and so on)

Machine learning
Shift from never ending if-else statements to ‘observing patterns'
Prediction accuracy has surpassed importance of ‘clearly definition’ of the distributions (at the cost of being able to directly explain “why”)

Three classes of machine learning
Supervised (studying labelled data, and then labelling new data)
	>> either classification (discrete, non-numeric variable) or regression (continuous variable)
	>> Does the person have the coronavirus?: Yes/No
Unsupervised learning
	>> creates models for data when no pre-existing labels on data
	>> grouping similar items, and making music/song recommendations
  3.  Reinforcement learning
	>> take certain actions - algorithm receives rewards for those actions
	>> self-driving cars and chess

Deep learning able to beat all prior machine learning algorithms. It can be applied in each of the above 3 classes of algorithm.
Barriers to deep learning:
it requires lots of data
it requires lots of processing power
difficult to understand such flexible algorithms.

Tensorflow and Scikit-learn are two of the most popular libraries for supervised/unsupervised learning.

Ethics of machine learning
Humans make biased decisions in today’s world
Data reflects these human biases
Left unchecked, machine learning methods will absorb the same biases

Linear Regression
Two tricks for line: y[_hat] = w1x + w2 and point = (p,q)
Absolute trick: move line closer to point as per modification
	y_hat = (w1 + pa)x + (w2 + a)     .. where point is above line, and x-value of point is positive
Square trick (equivalent to minimising ‘Mean squared error’)
	y_hat = (w1 + p(q - q’)a)x + (w2 + (q - q’)a)
n.b. Here y_hat is the (straight line) estimator for y

Mean absolute error:
E_mae = 1/m * sum(|yi - yi’|)
Mean squared error:
E_mse = 1/2m * sum([yi - yi’]^2)    (denominator increased by 2x for convenience)

Can examine error minimisation by taking derivative (wrt estimator parameters)
del mse / del w1 =   del [1/2 * (y - y_hat)2) ] / del w1
			   =   del [1/2 * (y - [w1x + w2])^2) ] / del w1
			   =   2 *  1/2 * (y - [w1x + w2]) * (-x)
			   =   -(y - y_hat) * x
And similarly..
de mse / del w2 =   -(y - y_hat) 

To actually conduct the increments:
wi --> wi - alpha * (del Error)/(del w1)
w1 --> w1 - alpha * (-(y - y_hat)x)
w2 --> w2 - alpha * (-(y - y_hat))

Variants of Gradient Descent (GD)
Can use derivative to move towards lower error (descend the error)
But how do we do it?
Batch GD
	Minimise E_mae = 1/m * sum(|yi - yi’|), by calculating contribution from all N points
Stochastic GD
calculate gradient from existing estimator (line) to the ideal line using just one point
then repeat, using the revised line and one new point.
repeat for lots of points until error minimisation is reached
Gives a noiser, but more random path towards minima
The ‘model’ update from each training example is described as an ‘online machine learning algorithm’
i.e. the model data is produced sequentially and used to update best predictor in the next step
Useful when fitting across entire dataset is computationally infeasible
Pros:
Requires less computation (if have 1million points, only use a tiny fraction [1/1000000] per round)
More globally optimum solution found if error manifold has lots of local minima (SGD can jerk routine of out local minima)
Cons:
Hard for algorithm to settle at minima
High variance in performance between each example (and can be difficult to explain)
Mini-Batch GD
	Splits training dataset into small batches, to calculate model error to update coefficients
	Routine sums the gradient over the mini-batch, thus reducing variance of the gradient
	Most recommended, pragmatic approach. Especially for deep learning.
	Pros
	Higher model update frequency than batch, therefore more robust at avoiding local minima
	Fewer re-fitting steps than stochastic can mean it can be more efficient than stochastic
	Cons
	More complex ~ requires ‘mini-batch size’ hyper parameter (plus both features of batch and stochastic designs)

Mini-Batch GD how to further look:
Batch sizes tuned to computational architecture so fits memory requirements of GPU/CPU.
Therefore consider the following sizes in batch: 32, 64, .. 512 etc.	
Often recommended: size = 32 points.
To decide more computationally: examine learning curves: time for training with different batch sizes



Docstring
Multi-line string not assigned to anything, specified in source code to specify code segment.
Docstrin specifies what a function does, not how.

Numpy

Documentation
Quick start guide: https://numpy.org/devdocs/user/quickstart.html


Basic object : ndarray.. homogeneous multi-dimensional array 
Features are
Fixed size at creation
Use vectorisation by default
Elements must be of same type (unless they’re python objects)

Broadcasting .. refers to element-by-element operations

Axes .. the dimensionality of an array
Rank .. the number of axes

Example: an ndarray consisting of 1,2,3 and also 4,5,6 has 2 elements
These look like:
A=	[[1,2,3],
	  [4,5,6]]
Therefore has dimension 2: like a 2-D matrix
	The first dimension has length 2 (I.e. number or rows)
	The second dimension has length 3 (number of cols)
Difference with standard python class array.array:
 - ndarrays are multi-dimensional, whereas array.array are 1-dimensional

Common ndarray attributes include:
ndarray.shape ... e.g. if 2 d matrix, returns (n [rows], m [cols]). Therefore length of shape is the dimension/rank of the ndarray
ndarray.size .. product of shape, e.g. if 2 d matrix: n x m
ndarray.ndim  ... number of dimensions. Also the rank.

np.arange(1,4) .. array of elements [1, 2, 3]    ..(excludes last)

np.zeros([2,3]) & nd.ones([2,3]) .. creates ndarray of zeros/ones with 2 rows and 3 columns

np.empty([2,3]) .. an array of random garbage attributes
	such as:
		array([[5.e-324, 1.e-323, 0.e+000],
			  [0.e+000, 2.e-323, 0.e+000]])
np.flip(nd.arange(1,4))  … reverse attributes, yields: [3,2,1]
np.nonzero(a) = np.where(a>0) .. index of non-zero elements

Size Attributes:
	Size:
np.array([1,34]).size   <- returns the size attribute of the object (not memory size)
np.size(a)  <- total number of elements
np.size(a,0) returns number of elements on axis 0 (vertical axis)
np.size(a,1) returns number of elements on axis 1 (horizontal axis)

	Itemsize:
np.itemsize(a) .. memory size of one (and any) element in np.array

np.diag(nd.array(..),k = 1) .. create array with your chosen diagonal, offset upwards by 1.

Double colon:
a = np.arange(100)
a[start:end:step]
a[10:20:3]  —> [10 13 16 19]
a[-10::-10] —> [90 80 70 .. 0]

Dimension
np.array(    [1,2,3]  ).shape      —> (3,)		    Not normally what we want
np.array(  [[1,2,3]] ).shape    —> (1,3)		Normally what we want (e.g. for scikit-learn)
	.. 1xN is default shape for numpy, i.e. np.random.randn(1,10).shape = (1,10)
np.array( [[[1,2,3]]] ).shape   —> (1,1,3)

Applying scalar functions can depend on axis provide 
np.mean() or np.std()
np.mean(a)			entire array
np.mean(a, axis = 0)	down each column
np.mean(a, axis = 1)	across each row
Example:
	a = np.arange(100).reshape([10,10])
	cols = np.mean(a,axis = 0)
	rows = np.mean(a,axis = 1)
	print('a = \n',a)
	print('Across columns = ',cols)
	print('Down rows = ',rows)

Slicing example:
	a = np.array(
	        [[1,0],
	         [0,2],
	         [0,0]]
	    )
	first_row  = a[0 ]
	second_col = a[0:3,1]
	print('first_row = \n',first_row)
	print('second_col = \n',second_col)
..returns:
	first_row = 
	 [1 0]
	second_col = 
	 [0 2 0]
Array-level comparisons, use ‘np.skyarray_equal()’ function
a = np.array([1,2,3])
b = np.array([1,2,3,4])
np.array_equal(a,b)

SCI-Kit Learn (scikit-learn)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
    — or — 
from sklearn import linear_model
model = LinearRegression()
model.fit(x,y)   <— this is a very general (widely used) fit command for scikit-learn

Once fitted, we have the following available:
model.coefficient_
model.intercept_
model.predict(123)

Load your classic datasets
from sklearn.datasets import load_iris
iris = load_iris()

Browse around the dataset
iris  ..or..  print(iris)
   .. this gives us clues as to what’s inside the object.. like ‘data’, ’target’, ‘feature/target_names’,’DESC’
Access each of these with:
print(iris['target_names'])

Bunch
Default type() of dataset from “from sklearn.datasets import load_dataset"
Can contain lots of different types of data / object
Object can be confirmed to be bunch, by using: type(iris)

Linear Regressions Closed Form Solution
Error minimisation of sum([y - y’]^2] wrt to w1 and w2 ..
.. gives a set of two equations, with two unknowns.
These can be substituted into one-another, and solved explicitly (hence “closed form”)
Question, what about closed form solutions in N dimension?
>> Well yes, closed form solutions can be attained via same method, but this..
    .. actually requires inverting an NxN matrix, which is computationally unfeasible.
    .. instead, gradient descent much more computationally feasible than seeking closed-form solution.

N-Dimensional LinearRegression closed form solution:
del E / del W = 0 = 2X^TXW-2X^Ty
This can be solved by W = (X^TX)^-1X^Ty

Weaknesses of LinearRegression
best represents data that is linear
sensitive to outliers

Polynomial Regression is similar to linear regression
for one dimensional case: y = w1 x X1^3 + w2 x X2^2 + w3 x X3 + w4
then minimisations ~~ del y / del wi   <— then solve simultaneously


