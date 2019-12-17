# Feature-map Aggregated Salient Object Detection
<br> Paul Vincent S. Nonat
<br> Mark Vincent Ty
<br> EE 298F - Advanced Deep Learning for Computer Vision
<br> Electrical and Electronics Engineering Institute
<br> University of the Philippines, Diliman

# Our Salient Object Detection Network
<br>Feature Aggregated Network
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/EE%20298F%20Final%20Project%20-%20Presentation%20(9).png)
# Loss Function
<br> Cross Entropy Loss
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/lce.gif)

<br>IoU Loss
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/liou.gif)

<br>Fussion Loss
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/lfuse.gif)

# For BASNet evaluation and structural architecture notebook, see
<br>https://github.com/paul028/EE298F_Final_Project/blob/master/BASNetStructEvaluation/evaluation_notebook.ipynb

# Evaluation Metric
<br>_
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/mae.gif)

# Experiment1 - CE
<br>a
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/Experiment%201Training%20Graph/Cross%20Entropy%20Loss%20Graph.png)
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/Experiment%201Training%20Graph/Average%20CE%20Loss%20per%20epoch.png)
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/Experiment%201Training%20Graph/MAE%20Graph.png)

# Experiment2 - IoU
<br>_
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/Experiment%201Training%20Graph/Average%20IoU%20per%20Epoch.png)
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/Experiment%201Training%20Graph/IoU%20Loss%20Graph.png)
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/Experiment%201Training%20Graph/MAE%20Graph.png)
# Qualitative Result
<br> Sample Results
![alt text](https://github.com/paul028/EE298F_Final_Project/blob/master/Experiment%201Training%20Graph/Result.gif)
# Comparison with Existing Models
<br> _
![alt text](https://github.com/paul028/EE298F_Final_Project/EvaluationResultsFinal.png)

# Dataset Used for Training and Evaluation
<br> Below are the Link for the dataset used in this model. Kindly site the mentioned paper if you inted to use these dataset.

<br>[DUTS Image Dataset](https://drive.google.com/file/d/1mdsP9Dq5e0C6US0h0HAajxfzhCWJZYHT/view?usp=sharing)

        @inproceedings{wang2017,
           author = {Wang, Lijun and Lu, Huchuan and Wang, Yifan and Feng, Mengyang and Wang, Dong and Yin, Baocai and Ruan, Xiang},
           title = {Learning to Detect Salient Objects With Image-Level Supervision},
           booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
           year = {2017}
        }

<br>[DUT-OMRON Image Dataset](https://drive.google.com/file/d/1iBT1jRlNKv7r3FfE2LsHSjJ1-MKt4qbi/view?usp=sharing)

        @inproceedings{yang2013saliency,
        title={Saliency detection via graph-based manifold ranking},
        author={Yang, Chuan and Zhang, Lihe and Lu, Huchuan, Ruan, Xiang and Yang, Ming-Hsuan},
        booktitle={Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference on},
        pages={3166--3173},
        year={2013},
        organization={IEEE}
        }

<br>[Extended Complex Scene Saliency Dataset (ECSSD)](https://drive.google.com/file/d/1kIUO6HyGFpHxRMwkCEJcHPlK-hmWLAWF/view?usp=sharing)
<br>[HKU-IS Dataset](https://drive.google.com/file/d/1RXFKl95yyyNmXnYJU35Y114KLzK4X0dZ/view?usp=sharing)
<br>[PASCAL-S](https://drive.google.com/file/d/1oaB4TYuozemKI9eqs6NK-9A8ED6t7OKz/view?usp=sharing)
<br>[SOD Dataset](https://drive.google.com/file/d/1Zj1yV1ILTkfidO7ABge57qA513zNZRyR/view?usp=sharing)

# Link to Presentation
<br>[You Tube Link](https://www.youtube.com/watch?v=aXTMk4uwJqU&feature=youtu.be&fbclid=IwAR1oA_XWo_d4PtcdujrbcHYEozQCq21g56YSpr-Fxolzk8AC-D8TtQXoCeM)
<br>[Slides](https://docs.google.com/presentation/d/1UZaVCmaVJl1sib-rPsstL9H0vRVqAuWsZi6HWPnZk8c/edit?usp=sharing)
