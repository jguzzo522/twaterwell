# Predicting Water-well Functionality Using Gradient Boosting Classifier
Water scarcity is a significant threat to human life in Tanzania. There are many water-wells in the country that try to address the issue of water starvation. However, the data indicates that many are non-functional. This project uses real world data provide by Tariff and the Tanzanian Ministry of water [Tanzanian Water](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/) , to help predict non-functional water-wells.

## Goal
The primary goal of this project is to identify factors contributing to the functionality status of water wells across Tanzania, distinguishing between functional and non-functional wells. The dependent variable studied was the 'status_group'. This binary classification was labeled as functional or non-functional water-wells. This project will provide significantly data to predict the likelihood of non functionality. The three major predictors of non-functioning water-wells are Dry Quantity of water, water point types labeled other, and older water-wells, especially water-wells created in the 1960's. This data allows the Tanzanian government, to mitigate potential water shortages.


![Screen Shot 2024-03-10 at 8 31 21 PM](https://github.com/jguzzo522/twaterwell/assets/75549456/2f50ea80-71c1-4820-8867-baec299bf0ed)

## Data

This project used existing data from a data set called ‘training_set_labels and training_set_values’. Pandas library was used to import this dataset into Jupyter notebook.

The original dataset can be found at https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/


| Column                | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| id                    | Unique identifier for each water point                        |
| amount_tsh            | Total static head (amount of water available at water point)  |
| date_recorded         | Date the information was entered into the database            |
| funder                | Organization that funded the well                             |
| gps_height            | Elevation of the water point                                  |
| installer             | Organization that installed the well                          |
| longitude             | Geographic longitude of the water point                       |
| latitude              | Geographic latitude of the water point                        |
| wpt_name              | Name of the water point                                       |
| num_private           | Private individuals or entities associated with the well      |
| basin                 | Geographic water basin                                        |
| subvillage            | Subvillage where the water point is located                   |
| region                | Geographic region                                             |
| region_code           | Coded form of the geographic region                           |
| district_code         | Coded form of the geographic district                         |
| lga                   | Local Government Area                                         |
| ward                  | Geographic ward                                               |
| population            | Estimated population using the water point                    |
| public_meeting        | If there was a public meeting about the water point           |
| recorded_by           | Entity that entered the data into the database                |
| scheme_management     | Organization managing the water scheme                        |
| scheme_name           | Name of the water scheme                                      |
| permit                | If the water point is permitted                               |
| construction_year     | Year the water point was constructed                          |
| extraction_type       | The kind of extraction the water point uses                   |
| extraction_type_group | Group classification of the extraction type                   |
| extraction_type_class | Class classification of the extraction type                   |
| management            | How the water point is managed                                |
| management_group      | Group classification of management                            |
| payment               | The payment structure for the water point                     |
| payment_type          | Type of payment                                               |
| water_quality         | Quality of the water                                          |
| quality_group         | Group classification of water quality                         |
| quantity              | The quantity of water available                               |
| quantity_group        | Group classification of water quantity                        |
| source                | The source of the water                                       |
| source_type           | Type of source                                                |
| source_class          | Class of water source                                         |
| waterpoint_type       | The kind of water point                                       |
| waterpoint_type_group | Group classification of the water point type                  |
| status_group          | The functionality status of the water point                   |

## Libraries Used

The analysis, modeling, manipulation, computation, visualization, and plotting of this data were facilitated by various Python library packages such as matplotlib, seaborn, pandas, numpy, and scikit-learn, including its metrics, model selection, impute, preprocessing, feature selection, pipeline, ensemble, linear model, and support vector machine modules, along with sys for system-specific parameters and functions, with additional packages utilized in the analysis and modeling process.

## Cleaning the Data

Two datasets were merged based on ID. The two datasets had similar or duplicate columns such as ‘quantity’ and ‘quantity_group’ were identified and removed. Following the removal of duplicate columns, further exploration of the data was conducted.

## Modification of the Target Variable

In order to simplify the analysis of the dependent variable 'status_group', I adjusted the Trinary classification into a Binary classification. 'Functional' and 'functional needs' repair', were merged into a single value, and 'non-functional' was left alone. 

## Removal of Non-Duplicate Columns


Before our modeling could begin, several columns were removed or changed to expedite that analysis :
- **Funder:** This column contained a large number of unique values (1897), making it impractical to analyze each individual funder. Removing this variable streamlines our analysis and allows us to focus on other potentially more impactful features.
- **Subvillage:** While the subvillage feature provides detailed geographic information, it also contains an excessive number of unique values (19,287). Given the large number of possible subvillages, this column was removed, to streamline the models.
- **Scheme Name:** Although both 'scheme_management' and 'scheme_name' represent the group responsible for operating the water-point, the latter contains a vast number of unique values (2696). We opted to retain the column 'scheme_management' variable for simplicity and ease of analysis.
- **Installer:** With 2146 unique values, the installer column indicates a wide variety of organizations or individuals responsible for installing water-points. Since analyzing each unique installer would be cumbersome we removed this variable to streamline the dataset.
- **Recorded By:** This column contains only one unique value, indicating that all records were recorded by the same entity. Since it does not offer any variability or meaningful information for analysis, removing it simplifies the dataset without sacrificing relevant information.
- **Ward and LGA:** These columns contained numerous individual values, while the district code provided sufficient geographical information. Therefore, we opted to retain the district code for geographic reference instead.
- 
## Handling Missing Data


### `construction_year
The 'constuction_year' column seemed logically important. As many buildings, or projects created many decades ago, often need repair. However, the data have
20,709 missing values. To address this issue, KNN (K-Nearest Neighbors) imputation was employed. This is a technique used in Data Science to address missing values by considering the values of similar neighbor data points rows. Then using the value in this case 'neighbors=5', the method analysis the similarities of columns, and replaces the unknown 'construction', with a best estimated value.  



### `Population`


The KNN imputation was also used for population. Keeping the `population` column is essential because its important to know how population effects the water-wells. Its also important to predict when an area with a high population will run out of water.

#### Addressing Categorical data

## Ordinal Encoding of Construction Decade:

OrdinalEncoder was used to create a chronological sequence of the construction decades, which allowed modeling to better analyze the data. Ordinal encoding transforms categorical data with a natural order into numerical values, preserving the rank or sequence inherent to the categories. This method enables algorithms to recognize and utilize the relative importance or progression of these categories, enhancing analytical and predictive processes.

The categories created were ordered by decade from 1960, 1970 , 1980 ,1990 ,2000, and 2010.

## Dummy Encoding of Categorical Columns
 To analyze categorical variables, the remaining columns including 'basin', 'region', 'extraction_type', 'management_group', 'payment', 'water_quality', 'quantity', 'source', 'waterpoint_type', , and 'gps_height' were dummy encoded using `pd.get_dummies()` function in pandas. 

Dummy encoding converts categorical variables into a set of binary variables. The categories are converted to 1 or 0, 1 indicating the presence and 0 indicating the absence of the category 

 ## Model 4: Gradient Boosting Classifier 

 ### Gradient Boosting Classifier Model Summary 
 In this analysis, the dataset underwent preprocessing, including one-hot encoding to transform non-numeric columns into a suitable format for model training. The dataset was then divided into training and testing sets using an 80-20 split, ensuring reproducibility with a `random_state` set to 42. After this a Gradient Boosting classifier was instantiated and trained on the encoded training data, followed by making predictions on the test dataset. 
 ## Model 4: Gradient Boosting Classifier
 
 ### Gradient Boosting Classifier Model Summary 
In this analysis, the dataset underwent preprocessing, including one-hot encoding to transform non-numeric columns into a suitable format for model training. The dataset was then divided into training and testing sets using an 80-20 split, ensuring reproducibility with a `random_state` set to 42. Subsequently, a Gradient Boosting classifier was instantiated and trained on the encoded training data, followed by making predictions on the test dataset.

### Model 4 Performancev

The model demonstrated an accuracy of approximately 79.81% on the test set, indicating its ability to correctly classify water-well functionality. Importantly, it also exhibited the lowest number of false negatives among all models, which is crucial for Tanzania's water distribution efforts. Achieving a high level of accuracy ensures reliable predictions, which are essential for effectively allocating resources and addressing water-related challenges. Minimizing false negatives is particularly important in this context, as it directly impacts human lives; access to clean water is a matter of life and death for communities in need.

![Screen Shot 2024-03-10 at 8 10 32 PM](https://github.com/jguzzo522/twaterwell/assets/75549456/ba75fec0-b85d-42a5-b4e4-be466e75a811)

#### Confusion Matrix 
The confusion matrix illustrates the model's predictions compared to the actual labels. It consists of four cells: 
 - **True Positive (TP)**: Water wells that are correctly classified as "Functional" (2529).
 - **False Positive (FP)**: Water wells that are incorrectly classified as "Non-Functional" when they are actually "Functional" (1897).
 - **False Negative (FN)**: Water wells that are incorrectly classified as "Functional" when they are actually "Non-Functional" (428).
 - **True Negative (TN)**: Water wells that are correctly classified as "Non-Functional" (6664). 

The confusion matrix can quickly show you where the model is fitted. For this project, it was important to reduce false negatives. A false negative could mean the difference of life and death, because these water-wells are non-functioning but the model indicate they are functional. This model does a fair job at eliminating this risk. 


### Classification Report

### ROC AUC Score 

This model has a ROC AUC score of around 0.756. This indicates the model has a strong ability to discriminate between positive and negative classes. This means this model is effective at making a distinction between function and non-functional water-wells

### Model Performance Metrics

#### Accuracy 
The accuracy metric measures the overall correctness of the model's predictions. In this case, the model achieved an accuracy of approximately 79.81%, meaning it correctly classified about 79.81% of all water wells. 

### Feature Importances for Gradient Boosting Classifier

According to this models data, quantity_dry', 'waterpoint_type_other', 'extraction_type_other', 'decade_construction_ordinal', were all important features. These variables were most important when predicting which water-wells functioned. 

## 'Water_quantity'

![Screen Shot 2024-03-10 at 8 18 26 PM](https://github.com/jguzzo522/twaterwell/assets/75549456/5de55cdc-1f41-48bc-933b-ae008cf02c47)

The first visualization highlights the critical role of 'water_quantity' in well functionality. Wells labeled as 'dry' are functional only about 5% of the time. Similarly, wells with an 'unknown' water quantity are functional in only around 25% of cases. The Tanzanian government should look to water-wells that are dry, to mitigate the water crisis.

## 'Water_point'

![Screen Shot 2024-03-10 at 8 18 01 PM](https://github.com/jguzzo522/twaterwell/assets/75549456/d57bc6a6-ed38-4dd6-9b2c-41cd261e5843)

This visualization shows the impact of water point types on well functionality. It highlights that water points categorized as 'other' have the lowest functionality rate, approximately 20%. Tanzania should focus its efforts on creating more water-wells that are communal standpipes(single).  This graph allows the government to visualize the problem, and potentially update the water point types when performing maintenance. The government should move away from 'other' type. water wells, and switch to more effective water points. 

# 'Decade_of_Construction' 

![Screen Shot 2024-03-10 at 8 17 32 PM](https://github.com/jguzzo522/twaterwell/assets/75549456/b6216fde-dd1c-41e5-abec-9c3d9358ccb3)

This visualization shows a clear trend: the older the well, particularly those constructed in the 1960s, 1970s, and 1980s, the less likely the well will function. In contrast, newer wells better performance. The government should allocate resources to renovating the old water-wells.

# Suggestions for the Tanzanian Government to Improve Water-well Production

The Tanzanian government should immediately prioritize interventions for water wells categorized as 'dry'. This classification shows that a large majority of dry water-wells are non-functional. The government should also address water points labeled other. These wells are most likely to be non functional. These wells should be redesigned to more successful types of water-points. The most successful being waterpoint_type_dam, and improved spring. If it is not feasible to adjust these waterpoints, at-least the government will know the locations at which water needs to be sent to. The government should also be looking at the decade in which the water-well was created. Functionality of water wells is significantly influenced by their age, with those constructed in the 1960s experiencing the highest rates of failure, underscoring the need for focused refurbishment efforts. Concentrated efforts to address these identified issues will not only enhance the reliability but also the accessibility of clean water resources. Adopting this strategic approach is paramount to fulfilling community water needs and achieving long-term sustainability in water resource management.

## Limitations

Although the dataset analyzed was comprehensive, it was not possible to examine every variable in detail. Notably, factors such as the well's funder or management entity, which could significantly impact water production, were not fully explored. For instance, cost-cutting measures by certain funders may compromise well quality, and some management companies may neglect necessary maintenance, impacting functionality. Without detailed analysis of these aspects, it's challenging to draw definitive conclusions about their impact on water well functionality. In addtion, without truly knowing the landscape like a local, it is impossible to know where building a dam would be possible. 

## Additional information

For a detailed analysis, see the [Tanwaterwell Notebook](https://github.com/jguzzo522/twaterwell/blob/main/tanwaterwell.ipynb).



