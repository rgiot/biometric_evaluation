Draft for a biometric evaluation library.
The code HAS NOT been extensively tested and MUST contain mistakes.

There are still a lot of work to do to allow other researchers to use this library.
And do not think I will do this work, so **feel free to fork and improve this project**.



Most of the code is a python implementation of the book `http://myslu.stlawu.edu/~msch/biometrics/`.

Various tools are available in the `tools` directory. 
They take as input text files formatted as required in the book `http://myslu.stlawu.edu/~msch/biometrics/`:

 * Column 1: gallery user
 * Column 2: probe user
 * Column 3: comparison number
 * Column 4: score

Library already allows to:

 * Compute EPC curves
 * Compute ROC curves
 * Compute Biometric Menagerie
 * Draw the score distribution

The tools are:

 * `EPC.py` allows to compare EPC curves
 * `ROC.py` allows to draw and compare ROC curves

Requirements:

  * Python 2.6
  * TraitsUI 4.2.0
  * Chaco 4.2.0 (Optionnal)
  * joblib
