# Reaffirming the Use of Satellite Imagery in Predicting Real Estate Prices
This repository contains data, code and trained models for my bachelor's thesis on "Reaffirming the Use of Satellite Imagery in Predicting Real Estate Prices". The objective of this thesis is the prediction of real estate prices in the city of New York (US), making use of traditional features such as square footage, as well as satellite images.

To run the code included in this repository, please ensure the packages listed in **requirements.txt** are installed in your current environment. You can run the following command to install all required packages:
```
pip install -r requirements.txt
```

The list of requirements is created using:
```
pip list --format=freeze > requirements.txt
```
Note: A significant amount of code has been moved to shared_functions.py to declutter the notebooks.

## Data Dictionary
| # | Column | Data Type | # Unique | # Missing | Variable Type |
| --- | --- | --- | --- | --- | --- |
0|borough|Int64|5|0|categorical
1|neighborhood|string|251|0|categorical
2|building_class_category|string|43|0|categorical
3|tax_class_at_present|string|10|93|categorical
4|block|Int64|9950|0|categorical
5|lot|Int64|2268|0|categorical
6|building_class_at_present|string|134|93|categorical
7|address|string|36993|0|categorical
8|apartment_number|string|3514|42280|categorical
9|zip_code|Int64|181|0|categorical
10|residential_units|Int64|71|18319|numerical
11|commercial_units|Int64|26|31291|numerical
12|total_units|Int64|75|16367|numerical
13|land_square_feet|Int64|4429|33243|numerical
14|gross_square_feet|Int64|4045|33243|numerical
15|year_built|Int64|168|3452|numerical
16|tax_class_at_time_of_sale|Int64|3|0|categorical
17|building_class_at_time_of_sale|string|134|0|categorical
18|sale_price|Int64|7988|0|numerical
19|sale_date|Int64|321|0|numerical
20|location_lat|Float64|34478|0|numerical
21|location_long|Float64|34531|0|numerical
22|tax_class_at_present_prefix|string|3|93|categorical
23|building_class_at_present_prefix|string|22|93|categorical
24|street_name|string|5825|16095|categorical
25|building_class_at_time_of_sale_prefix|string|22|0|categorical
26|sale_price_adj|Float64|18639|0|numerical

Can be found under: data/processed/property-sales_new-york-city_2022_pre-processed_data-overview.csv