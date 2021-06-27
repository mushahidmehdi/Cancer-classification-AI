# Classification of benign and malignant cells AI Model


With the help of computer vision, Computer-aided diagnosis (CAD) has become a part of the routine clinical work for detection of cancer. This seems to indicate that CAD is beginning to be applied widely in the differential diagnosis and detection of many types of abnormalities in medical images obtained in various examinations by use of different imaging modalities. In fact, CAD has become one of the major research subjects in medical imaging and diagnostic radiology. 

![CAD-acts-as-a-mediator-connecting-medicine-with-computer-science-20](https://user-images.githubusercontent.com/66418035/119219085-4b1a6000-baec-11eb-9981-757fe7cd9e87.png)

Having said that I would like to explain what we are about to do in this project. First let's get some medical terminoligies straight. 

Benign: Benign is a tumors are not harmful in most cases, and they are unlikely to affect other parts of the body.

Malignant: cancerous tumors, develop when cells grow uncontrollably. If the cells continue to grow and spread, this can become life threatening.
![Benign-and-malignant-melanoma](https://user-images.githubusercontent.com/66418035/119219094-5a011280-baec-11eb-9b2e-d5eace536a46.png)

![malignant-and-benign-tumor-vector-26703501](https://user-images.githubusercontent.com/66418035/119219047-20300c00-baec-11eb-892a-bbf017057976.jpg)

First of all lets have a look of what kind of data we have, and what can we make out of it. There are 500k of medical images avalibale from diffent
resources along with their CSV files. The Good News is data in not that much messay as usually they are, so we will compeletly take advantage of that.
we will map the CVS file along with its respective images and fit into the model, so we can check what results it generates,by comparing with
the label data from CVS file.

The model weights and parameters are saved, as I was using Pychram, therefore, there wouldn't be any data visualizaition. Honestly,
we don't need any visualization, as there is no data manipulation involved here.

To load the the saved model without python code just add folowing code in your IDE

`model.load('my_model')`


You can also train and validate a new model on same data, you can download all the required materials on links below:

[Training Images](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip)

[Training CSV Features](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv)

[Training CSV GroudData](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv)

[Test Dataset](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip)


