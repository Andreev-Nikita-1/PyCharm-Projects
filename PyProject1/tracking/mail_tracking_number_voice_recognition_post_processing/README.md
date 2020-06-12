This package allows you to convert CTC-trained Neural Networks output into mail tracking number in *International mail* or *Russian Post* format  


Format example for *International mail*: ``RO260964943RU``  
Format example for *Russian Post*: ``1421171600738``

Class ``TrackingNumberRecognizer`` has method ``convert(matrix: np.array) -> list``, which takes 2d matrix and returns list of possible tracking numbers in decreasing order of probability


Matrix first dimension corresponds to the time moment. Second dimension corresponds to a letter, last index has to correspond to *blank* (see CTC).
**Russian letters supported only**.

Value is the probability of pronunciation letter at the moment.

Format (*International mail* or *Russian Post*) is set in  ``TrackingNumberRecognizer`` constructor.

You can also set the letters order, the probability of your model making mistake, and the prior probabilities for letters in case of *International mail* format 
 

