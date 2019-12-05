# Breast-Cancer-Diagnostic

This is a classification model which uses logistic regression or linear regression to determine whether the tumor in a womans breast
is a <strong>BENIGN</strong> or <strong>MALIGNANT</strong> tumor. The datasets that i used is the UCI [Breast Cancer Wisconsin (Diagnostic) Data Set ](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) 

<h3>How the input is taken and processed to ouput</h3>

<p><strong><h2>Input</h2></strong></p>
The inputs are a bunch of features and measurements of the tumor like the measurement of the Nuclei which includes measuring
<ul>
  <li>Radius</li>
  <li>Texture</li>
  <li>Perimeter</li> 
  <li>Area</li>
  </ul>
  and more. 
 
 <p><strong><h2> Ouput </h2></strong></p>
  The ouput is the classification. The dataset came as M for MALIGNANT and B for BENIGN. In the datasets, i exchange M for 1 and B for 0
  and i divided the dataset into two that is (X_data.csv) and (Y_data.csv) 
  
  Every other thing about the model is found in this [notebook](https://github.com/Mbah-Javis/Breast-Cancer-Diagnostic/blob/master/Classification%20Model%20on%20Breast%20Cancer%20Data%20set%20.ipynb) that is i commented on every line of code.
  
