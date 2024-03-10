# Predicting Water-well Functionality based on Analysis from Model 4: Gradient Boosting Classifier
Water scarcity poses a significant challenge in Tanzania, impacting various aspects of life. This project endeavors to predict non-functioning water wells to aid in addressing this critical issue.
## Goal
The primary goal of this project is to identify factors contributing to the functionality status of water wells across Tanzania, distinguishing between functional and non-functional wells. Utilizing the 'status_group' variable as an indicator of functionality, our analysis indicates that three key categories significantly influence the likelihood of non functionality:
	Quantity of Water - Dry: Water wells with dry water sources exhibit a higher probability of being non-functional.
	Water Point Type - Other: Water wells categorized as 'other' water point types are more prone to being non-functional.
	Decade of Creation: Older water wells, particularly those established in the 1960s, demonstrate a heightened risk of being non-functional.

Identifying potential points of failure across these categories is crucial for proactive intervention by the Tanzanian government to mitigate water distribution challenges and ensure access to clean water for the population.

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
