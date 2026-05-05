Consumer Behavior & Shopping Habits Analysis for Targeted Marketing Strategies
1. Problem Statement
Investigate the impact of various factors, such as seasonality, item attributes (size and color), and promotional activities (discounts and promo codes), on customer purchase decisions. Are there particular seasons when certain product categories perform better? Do specific item attributes or promotions significantly influence purchase amounts and review ratings?

Using the Consumer Behavior and Shopping Habits Dataset, analyze the relationship between customer demographics (age and gender) and their purchase behavior. Are there any specific product categories or shopping channels that are preferred by certain age groups or genders? How can this information be leveraged to design more targeted marketing strategies?

2. Importing Librearires
Importing Librearires
import numpy as np 
import pandas as pd 
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[1], line 4
      1 import numpy as np
      2 import pandas as pd
      3 import seaborn as sns
----> 4 import plotly.express as px
      5 import matplotlib.pyplot as plt
      6 from matplotlib import style
      7 get_ipython().run_line_magic('matplotlib', 'inline')

ModuleNotFoundError: No module named 'plotly'
df = pd.read_csv('/kaggle/input/consumer-behavior-and-shopping-habits-dataset/shopping_behavior_updated.csv')
3. Data Prepration and Cleaning
Data Prepration and Cleaning
df.head(5)
Customer ID	Age	Gender	Item Purchased	Category	Purchase Amount (USD)	Location	Size	Color	Season	Review Rating	Subscription Status	Shipping Type	Discount Applied	Promo Code Used	Previous Purchases	Payment Method	Frequency of Purchases
0	1	55	Male	Blouse	Clothing	53	Kentucky	L	Gray	Winter	3.1	Yes	Express	Yes	Yes	14	Venmo	Fortnightly
1	2	19	Male	Sweater	Clothing	64	Maine	L	Maroon	Winter	3.1	Yes	Express	Yes	Yes	2	Cash	Fortnightly
2	3	50	Male	Jeans	Clothing	73	Massachusetts	S	Maroon	Spring	3.1	Yes	Free Shipping	Yes	Yes	23	Credit Card	Weekly
3	4	21	Male	Sandals	Footwear	90	Rhode Island	M	Maroon	Spring	3.5	Yes	Next Day Air	Yes	Yes	49	PayPal	Weekly
4	5	45	Male	Blouse	Clothing	49	Oregon	M	Turquoise	Spring	2.7	Yes	Free Shipping	Yes	Yes	31	PayPal	Annually
df.tail(5)
Customer ID	Age	Gender	Item Purchased	Category	Purchase Amount (USD)	Location	Size	Color	Season	Review Rating	Subscription Status	Shipping Type	Discount Applied	Promo Code Used	Previous Purchases	Payment Method	Frequency of Purchases
3895	3896	40	Female	Hoodie	Clothing	28	Virginia	L	Turquoise	Summer	4.2	No	2-Day Shipping	No	No	32	Venmo	Weekly
3896	3897	52	Female	Backpack	Accessories	49	Iowa	L	White	Spring	4.5	No	Store Pickup	No	No	41	Bank Transfer	Bi-Weekly
3897	3898	46	Female	Belt	Accessories	33	New Jersey	L	Green	Spring	2.9	No	Standard	No	No	24	Venmo	Quarterly
3898	3899	44	Female	Shoes	Footwear	77	Minnesota	S	Brown	Summer	3.8	No	Express	No	No	24	Venmo	Weekly
3899	3900	52	Female	Handbag	Accessories	81	California	M	Beige	Spring	3.1	No	Store Pickup	No	No	33	Venmo	Quarterly
df.shape
(3900, 18)
df.columns
Index(['Customer ID', 'Age', 'Gender', 'Item Purchased', 'Category',
       'Purchase Amount (USD)', 'Location', 'Size', 'Color', 'Season',
       'Review Rating', 'Subscription Status', 'Shipping Type',
       'Discount Applied', 'Promo Code Used', 'Previous Purchases',
       'Payment Method', 'Frequency of Purchases'],
      dtype='object')
It is useful to first understand the data that you’re dealing with along with the data types of each column. The data types may dictate how you transform and engineer features
#Data reading, checking dimensions and information of the data
print(df)

print('dimensions:')
print(df.shape)

print('Information:')
df.info()
      Customer ID  Age  Gender Item Purchased     Category  \
0               1   55    Male         Blouse     Clothing   
1               2   19    Male        Sweater     Clothing   
2               3   50    Male          Jeans     Clothing   
3               4   21    Male        Sandals     Footwear   
4               5   45    Male         Blouse     Clothing   
...           ...  ...     ...            ...          ...   
3895         3896   40  Female         Hoodie     Clothing   
3896         3897   52  Female       Backpack  Accessories   
3897         3898   46  Female           Belt  Accessories   
3898         3899   44  Female          Shoes     Footwear   
3899         3900   52  Female        Handbag  Accessories   

      Purchase Amount (USD)       Location Size      Color  Season  \
0                        53       Kentucky    L       Gray  Winter   
1                        64          Maine    L     Maroon  Winter   
2                        73  Massachusetts    S     Maroon  Spring   
3                        90   Rhode Island    M     Maroon  Spring   
4                        49         Oregon    M  Turquoise  Spring   
...                     ...            ...  ...        ...     ...   
3895                     28       Virginia    L  Turquoise  Summer   
3896                     49           Iowa    L      White  Spring   
3897                     33     New Jersey    L      Green  Spring   
3898                     77      Minnesota    S      Brown  Summer   
3899                     81     California    M      Beige  Spring   

      Review Rating Subscription Status   Shipping Type Discount Applied  \
0               3.1                 Yes         Express              Yes   
1               3.1                 Yes         Express              Yes   
2               3.1                 Yes   Free Shipping              Yes   
3               3.5                 Yes    Next Day Air              Yes   
4               2.7                 Yes   Free Shipping              Yes   
...             ...                 ...             ...              ...   
3895            4.2                  No  2-Day Shipping               No   
3896            4.5                  No    Store Pickup               No   
3897            2.9                  No        Standard               No   
3898            3.8                  No         Express               No   
3899            3.1                  No    Store Pickup               No   

     Promo Code Used  Previous Purchases Payment Method Frequency of Purchases  
0                Yes                  14          Venmo            Fortnightly  
1                Yes                   2           Cash            Fortnightly  
2                Yes                  23    Credit Card                 Weekly  
3                Yes                  49         PayPal                 Weekly  
4                Yes                  31         PayPal               Annually  
...              ...                 ...            ...                    ...  
3895              No                  32          Venmo                 Weekly  
3896              No                  41  Bank Transfer              Bi-Weekly  
3897              No                  24          Venmo              Quarterly  
3898              No                  24          Venmo                 Weekly  
3899              No                  33          Venmo              Quarterly  

[3900 rows x 18 columns]
dimensions:
(3900, 18)
Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3900 entries, 0 to 3899
Data columns (total 18 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   Customer ID             3900 non-null   int64  
 1   Age                     3900 non-null   int64  
 2   Gender                  3900 non-null   object 
 3   Item Purchased          3900 non-null   object 
 4   Category                3900 non-null   object 
 5   Purchase Amount (USD)   3900 non-null   int64  
 6   Location                3900 non-null   object 
 7   Size                    3900 non-null   object 
 8   Color                   3900 non-null   object 
 9   Season                  3900 non-null   object 
 10  Review Rating           3900 non-null   float64
 11  Subscription Status     3900 non-null   object 
 12  Shipping Type           3900 non-null   object 
 13  Discount Applied        3900 non-null   object 
 14  Promo Code Used         3900 non-null   object 
 15  Previous Purchases      3900 non-null   int64  
 16  Payment Method          3900 non-null   object 
 17  Frequency of Purchases  3900 non-null   object 
dtypes: float64(1), int64(4), object(13)
memory usage: 548.6+ KB
print(df.apply(lambda col: col.unique())) 
Customer ID               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...
Age                       [55, 19, 50, 21, 45, 46, 63, 27, 26, 57, 53, 3...
Gender                                                       [Male, Female]
Item Purchased            [Blouse, Sweater, Jeans, Sandals, Sneakers, Sh...
Category                       [Clothing, Footwear, Outerwear, Accessories]
Purchase Amount (USD)     [53, 64, 73, 90, 49, 20, 85, 34, 97, 31, 68, 7...
Location                  [Kentucky, Maine, Massachusetts, Rhode Island,...
Size                                                          [L, S, M, XL]
Color                     [Gray, Maroon, Turquoise, White, Charcoal, Sil...
Season                                       [Winter, Spring, Summer, Fall]
Review Rating             [3.1, 3.5, 2.7, 2.9, 3.2, 2.6, 4.8, 4.1, 4.9, ...
Subscription Status                                               [Yes, No]
Shipping Type             [Express, Free Shipping, Next Day Air, Standar...
Discount Applied                                                  [Yes, No]
Promo Code Used                                                   [Yes, No]
Previous Purchases        [14, 2, 23, 49, 31, 19, 8, 4, 26, 10, 37, 34, ...
Payment Method            [Venmo, Cash, Credit Card, PayPal, Bank Transf...
Frequency of Purchases    [Fortnightly, Weekly, Annually, Quarterly, Bi-...
dtype: object
df.nunique()
Customer ID               3900
Age                         53
Gender                       2
Item Purchased              25
Category                     4
Purchase Amount (USD)       81
Location                    50
Size                         4
Color                       25
Season                       4
Review Rating               26
Subscription Status          2
Shipping Type                6
Discount Applied             2
Promo Code Used              2
Previous Purchases          50
Payment Method               6
Frequency of Purchases       7
dtype: int64
df.isnull().sum()
Customer ID               0
Age                       0
Gender                    0
Item Purchased            0
Category                  0
Purchase Amount (USD)     0
Location                  0
Size                      0
Color                     0
Season                    0
Review Rating             0
Subscription Status       0
Shipping Type             0
Discount Applied          0
Promo Code Used           0
Previous Purchases        0
Payment Method            0
Frequency of Purchases    0
dtype: int64
#Cheking for duplicates 
value=len(df[df.duplicated()])
print(value) 
0
df.describe().T
count	mean	std	min	25%	50%	75%	max
Customer ID	3900.0	1950.500000	1125.977353	1.0	975.75	1950.5	2925.25	3900.0
Age	3900.0	44.068462	15.207589	18.0	31.00	44.0	57.00	70.0
Purchase Amount (USD)	3900.0	59.764359	23.685392	20.0	39.00	60.0	81.00	100.0
Review Rating	3900.0	3.749949	0.716223	2.5	3.10	3.7	4.40	5.0
Previous Purchases	3900.0	25.351538	14.447125	1.0	13.00	25.0	38.00	50.0
location_counts = df["Location"].value_counts()
print("Location Counts:\n", location_counts)
Location Counts:
 Location
Montana           96
California        95
Idaho             93
Illinois          92
Alabama           89
Minnesota         88
Nebraska          87
New York          87
Nevada            87
Maryland          86
Delaware          86
Vermont           85
Louisiana         84
North Dakota      83
Missouri          81
West Virginia     81
New Mexico        81
Mississippi       80
Indiana           79
Georgia           79
Kentucky          79
Arkansas          79
North Carolina    78
Connecticut       78
Virginia          77
Ohio              77
Tennessee         77
Texas             77
Maine             77
South Carolina    76
Colorado          75
Oklahoma          75
Wisconsin         75
Oregon            74
Pennsylvania      74
Washington        73
Michigan          73
Alaska            72
Massachusetts     72
Wyoming           71
Utah              71
New Hampshire     71
South Dakota      70
Iowa              69
Florida           68
New Jersey        67
Hawaii            65
Arizona           65
Kansas            63
Rhode Island      63
Name: count, dtype: int64
category_counts = df.groupby("Location")["Category"].value_counts()
print("Regional Category Trends:\n", category_counts)
Regional Category Trends:
 Location   Category   
Alabama    Clothing       41
           Accessories    25
           Footwear       15
           Outerwear       8
Alaska     Clothing       33
                          ..
Wisconsin  Outerwear       3
Wyoming    Clothing       31
           Accessories    23
           Footwear       11
           Outerwear       6
Name: count, Length: 200, dtype: int64
location_purchase_stats = df.groupby("Location")["Purchase Amount (USD)"].agg(["mean", "median", "sum"])
print("Regional Purchase Amount Stats:\n", location_purchase_stats)
Regional Purchase Amount Stats:
                      mean  median   sum
Location                               
Alabama         59.112360    56.0  5261
Alaska          67.597222    68.5  4867
Arizona         66.553846    68.0  4326
Arkansas        61.113924    58.0  4828
California      59.000000    57.0  5605
Colorado        56.293333    51.0  4222
Connecticut     54.179487    48.5  4226
Delaware        55.325581    52.5  4758
Florida         55.852941    56.0  3798
Georgia         58.797468    62.0  4645
Hawaii          57.723077    55.0  3752
Idaho           60.075269    62.0  5587
Illinois        61.054348    65.0  5617
Indiana         58.924051    60.0  4655
Iowa            60.884058    60.0  4201
Kansas          54.555556    50.0  3437
Kentucky        55.721519    53.0  4402
Louisiana       57.714286    55.5  4848
Maine           56.987013    57.0  4388
Maryland        55.755814    52.0  4795
Massachusetts   60.888889    64.0  4384
Michigan        62.095890    63.0  4533
Minnesota       56.556818    53.0  4977
Mississippi     61.037500    64.0  4883
Missouri        57.913580    61.0  4691
Montana         60.250000    64.0  5784
Nebraska        59.448276    58.0  5172
Nevada          63.379310    66.0  5514
New Hampshire   59.422535    59.0  4219
New Jersey      56.746269    51.0  3802
New Mexico      61.901235    63.0  5014
New York        60.425287    62.0  5257
North Carolina  60.794872    62.5  4742
North Dakota    62.891566    64.0  5220
Ohio            60.376623    66.0  4649
Oklahoma        58.346667    60.0  4376
Oregon          57.337838    54.0  4243
Pennsylvania    66.567568    70.0  4926
Rhode Island    61.444444    63.0  3871
South Carolina  58.407895    59.0  4439
South Dakota    60.514286    62.0  4236
Tennessee       61.974026    61.0  4772
Texas           61.194805    62.0  4712
Utah            62.577465    65.0  4443
Vermont         57.176471    54.0  4860
Virginia        62.883117    62.0  4842
Washington      63.328767    63.0  4623
West Virginia   63.876543    66.0  5174
Wisconsin       55.946667    55.0  4196
Wyoming         60.690141    60.0  4309
shipping_type_counts = df.groupby("Location")["Shipping Type"].value_counts()
print("Regional Shipping Type Trends:\n", shipping_type_counts)
Regional Shipping Type Trends:
 Location  Shipping Type 
Alabama   Express           20
          Store Pickup      19
          Next Day Air      17
          2-Day Shipping    16
          Free Shipping      9
                            ..
Wyoming   Express           13
          Next Day Air      12
          Standard          11
          Free Shipping     10
          2-Day Shipping     8
Name: count, Length: 300, dtype: int64
location_groups = df.groupby("Location")

# Analyze regional trends
for location, location_data in location_groups:
    print(f"Regional Trends for {location}:")

    # Calculate average purchase amount in this region
    avg_purchase_amount = location_data["Purchase Amount (USD)"].mean()
    print(f"Average Purchase Amount: ${avg_purchase_amount:.2f}")

    # Count the most popular product categories in this region
    popular_categories = location_data["Category"].value_counts().idxmax()
    print(f"Most Popular Category: {popular_categories}")

    # Analyze online shopping preferences
    online_shopping = location_data["Shipping Type"].apply(lambda x: "Online" if "Express" in x or "Standard" in x else "Offline")
    online_percentage = (online_shopping.value_counts() / len(online_shopping)) * 100
    print(f"Online Shopping Preference:")
    print(online_percentage)

    # Consider other factors based on your data and business context

    print("\n")
Regional Trends for Alabama:
Average Purchase Amount: $59.11
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    68.539326
Online     31.460674
Name: count, dtype: float64


Regional Trends for Alaska:
Average Purchase Amount: $67.60
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    58.333333
Online     41.666667
Name: count, dtype: float64


Regional Trends for Arizona:
Average Purchase Amount: $66.55
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    66.153846
Online     33.846154
Name: count, dtype: float64


Regional Trends for Arkansas:
Average Purchase Amount: $61.11
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    65.822785
Online     34.177215
Name: count, dtype: float64


Regional Trends for California:
Average Purchase Amount: $59.00
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    66.315789
Online     33.684211
Name: count, dtype: float64


Regional Trends for Colorado:
Average Purchase Amount: $56.29
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    76.0
Online     24.0
Name: count, dtype: float64


Regional Trends for Connecticut:
Average Purchase Amount: $54.18
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    70.512821
Online     29.487179
Name: count, dtype: float64


Regional Trends for Delaware:
Average Purchase Amount: $55.33
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    73.255814
Online     26.744186
Name: count, dtype: float64


Regional Trends for Florida:
Average Purchase Amount: $55.85
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    60.294118
Online     39.705882
Name: count, dtype: float64


Regional Trends for Georgia:
Average Purchase Amount: $58.80
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    73.417722
Online     26.582278
Name: count, dtype: float64


Regional Trends for Hawaii:
Average Purchase Amount: $57.72
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    64.615385
Online     35.384615
Name: count, dtype: float64


Regional Trends for Idaho:
Average Purchase Amount: $60.08
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    69.892473
Online     30.107527
Name: count, dtype: float64


Regional Trends for Illinois:
Average Purchase Amount: $61.05
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    65.217391
Online     34.782609
Name: count, dtype: float64


Regional Trends for Indiana:
Average Purchase Amount: $58.92
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    69.620253
Online     30.379747
Name: count, dtype: float64


Regional Trends for Iowa:
Average Purchase Amount: $60.88
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    62.318841
Online     37.681159
Name: count, dtype: float64


Regional Trends for Kansas:
Average Purchase Amount: $54.56
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    65.079365
Online     34.920635
Name: count, dtype: float64


Regional Trends for Kentucky:
Average Purchase Amount: $55.72
Most Popular Category: Accessories
Online Shopping Preference:
Shipping Type
Offline    58.227848
Online     41.772152
Name: count, dtype: float64


Regional Trends for Louisiana:
Average Purchase Amount: $57.71
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    67.857143
Online     32.142857
Name: count, dtype: float64


Regional Trends for Maine:
Average Purchase Amount: $56.99
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    61.038961
Online     38.961039
Name: count, dtype: float64


Regional Trends for Maryland:
Average Purchase Amount: $55.76
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    63.953488
Online     36.046512
Name: count, dtype: float64


Regional Trends for Massachusetts:
Average Purchase Amount: $60.89
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    72.222222
Online     27.777778
Name: count, dtype: float64


Regional Trends for Michigan:
Average Purchase Amount: $62.10
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    64.383562
Online     35.616438
Name: count, dtype: float64


Regional Trends for Minnesota:
Average Purchase Amount: $56.56
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    64.772727
Online     35.227273
Name: count, dtype: float64


Regional Trends for Mississippi:
Average Purchase Amount: $61.04
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    73.75
Online     26.25
Name: count, dtype: float64


Regional Trends for Missouri:
Average Purchase Amount: $57.91
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    56.790123
Online     43.209877
Name: count, dtype: float64


Regional Trends for Montana:
Average Purchase Amount: $60.25
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    75.0
Online     25.0
Name: count, dtype: float64


Regional Trends for Nebraska:
Average Purchase Amount: $59.45
Most Popular Category: Accessories
Online Shopping Preference:
Shipping Type
Offline    63.218391
Online     36.781609
Name: count, dtype: float64


Regional Trends for Nevada:
Average Purchase Amount: $63.38
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    68.965517
Online     31.034483
Name: count, dtype: float64


Regional Trends for New Hampshire:
Average Purchase Amount: $59.42
Most Popular Category: Accessories
Online Shopping Preference:
Shipping Type
Offline    69.014085
Online     30.985915
Name: count, dtype: float64


Regional Trends for New Jersey:
Average Purchase Amount: $56.75
Most Popular Category: Accessories
Online Shopping Preference:
Shipping Type
Offline    64.179104
Online     35.820896
Name: count, dtype: float64


Regional Trends for New Mexico:
Average Purchase Amount: $61.90
Most Popular Category: Accessories
Online Shopping Preference:
Shipping Type
Offline    69.135802
Online     30.864198
Name: count, dtype: float64


Regional Trends for New York:
Average Purchase Amount: $60.43
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    67.816092
Online     32.183908
Name: count, dtype: float64


Regional Trends for North Carolina:
Average Purchase Amount: $60.79
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    73.076923
Online     26.923077
Name: count, dtype: float64


Regional Trends for North Dakota:
Average Purchase Amount: $62.89
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    62.650602
Online     37.349398
Name: count, dtype: float64


Regional Trends for Ohio:
Average Purchase Amount: $60.38
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    76.623377
Online     23.376623
Name: count, dtype: float64


Regional Trends for Oklahoma:
Average Purchase Amount: $58.35
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    60.0
Online     40.0
Name: count, dtype: float64


Regional Trends for Oregon:
Average Purchase Amount: $57.34
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    67.567568
Online     32.432432
Name: count, dtype: float64


Regional Trends for Pennsylvania:
Average Purchase Amount: $66.57
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    68.918919
Online     31.081081
Name: count, dtype: float64


Regional Trends for Rhode Island:
Average Purchase Amount: $61.44
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    58.730159
Online     41.269841
Name: count, dtype: float64


Regional Trends for South Carolina:
Average Purchase Amount: $58.41
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    63.157895
Online     36.842105
Name: count, dtype: float64


Regional Trends for South Dakota:
Average Purchase Amount: $60.51
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    62.857143
Online     37.142857
Name: count, dtype: float64


Regional Trends for Tennessee:
Average Purchase Amount: $61.97
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    70.12987
Online     29.87013
Name: count, dtype: float64


Regional Trends for Texas:
Average Purchase Amount: $61.19
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    68.831169
Online     31.168831
Name: count, dtype: float64


Regional Trends for Utah:
Average Purchase Amount: $62.58
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    64.788732
Online     35.211268
Name: count, dtype: float64


Regional Trends for Vermont:
Average Purchase Amount: $57.18
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    63.529412
Online     36.470588
Name: count, dtype: float64


Regional Trends for Virginia:
Average Purchase Amount: $62.88
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    55.844156
Online     44.155844
Name: count, dtype: float64


Regional Trends for Washington:
Average Purchase Amount: $63.33
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    71.232877
Online     28.767123
Name: count, dtype: float64


Regional Trends for West Virginia:
Average Purchase Amount: $63.88
Most Popular Category: Accessories
Online Shopping Preference:
Shipping Type
Offline    72.839506
Online     27.160494
Name: count, dtype: float64


Regional Trends for Wisconsin:
Average Purchase Amount: $55.95
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    64.0
Online     36.0
Name: count, dtype: float64


Regional Trends for Wyoming:
Average Purchase Amount: $60.69
Most Popular Category: Clothing
Online Shopping Preference:
Shipping Type
Offline    66.197183
Online     33.802817
Name: count, dtype: float64


In Alaska, shoppers display a strong inclination for clothing, with an average spending of $67, which is the highest among states in the Clothing category. This suggests a distinct preference for quality and style, reflecting unique consumer behavior and market trends in Alaska.
4. Data Visualization
Data Visualization
Let's have look via visulaization into the impact of various factors, such as seasonality, item attributes (size and color), and promotional activities (discounts and promo codes), on customer purchase decisions.
seasons = df['Season'].unique()
average_purchase_by_season = df.groupby('Season')['Purchase Amount (USD)'].mean()

plt.figure(figsize=(6, 4))
plt.bar(seasons, average_purchase_by_season, color=['skyblue', 'lightcoral', 'lightgreen', 'lightpink'])
plt.title("Impact of Season on Purchase")
plt.xlabel("Season")
plt.ylabel("Average Purchase (USD)")
plt.show()

We can see from the visual that customers bought more in the winter and autumn than they did in the spring and summer.
plt.figure(figsize=(6, 4))
sns.barplot(x='Category', y='Purchase Amount (USD)', data=df, ci=None, palette='viridis')
plt.title("Impact of Category on Purchase")
plt.xticks(rotation=45)
plt.show()

The graph shows that the Outerwear Category is a bit lower than the other categories.
gender_purchase = df.groupby('Gender')['Purchase Amount (USD)'].sum()

# Create a donut chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.pie(gender_purchase, labels=gender_purchase.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'yellow'], wedgeprops=dict(width=0.4))
ax.set_title("Impact of Gender on Purchase")
plt.axis('equal')  # Equal aspect ratio ensures that the chart is drawn as a circle.

# Draw a circle in the center to create a donut chart
center_circle = plt.Circle((0,0),0.70,fc='white')
fig.gca().add_artist(center_circle)

plt.show()

Males are more likely to spend (67%), whereas females are less likely to spend (32%).
plt.figure(figsize=(6, 4))
sns.swarmplot(x='Size', y='Purchase Amount (USD)', data=df, palette='Set2')
plt.title("Impact of Size on Purchase")
plt.xlabel('Size')
plt.ylabel('Purchase Amount (USD)')
plt.xticks(rotation=45)
plt.show()

As shown in the chart, the XL size had less purchases than the other sizes Large, Small, and Mediam.
promo_counts = df['Promo Code Used'].value_counts()

# Create a pie chart
plt.figure(figsize=(6, 4))
plt.pie(promo_counts, labels=promo_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral'])
plt.title("Impact of Promo Code Used on Purchase")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

As we can see there is no such imapacte on using Promocode on purchase.
Let's analyze the relationship between customer location (age and gender) and their purchase behavior.
location_counts.plot(kind="bar", figsize=(12, 4))
plt.title("Customer Distribution by Location")
plt.xlabel("Location")
plt.ylabel("Number of Customers")
plt.show()

Montana stands out with its remarkable number of customers, surpassing all other states in this regard. The state's thriving business landscape and vibrant consumer market have contributed to its impressive customer base.
category_counts = df['Category'].value_counts()

# Define a list of different colors for each bar
colors = ['skyblue', 'lightcoral', 'lightseagreen', 'lightsalmon', 'lightpink']

# Create a figure and axis
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot the bar chart with different colors for each bar
bars = plt.bar(category_counts.index, category_counts.values, color=colors)

# Add labels and title
plt.xlabel('Product Categories')
plt.ylabel('Count')
plt.title('Distribution of Product Categories')
plt.xticks(rotation=90)

# Display the chart
plt.tight_layout()

# Optionally, you can add a legend to show the correspondence between colors and categories
legend_labels = category_counts.index[:len(colors)]  # Take labels for the number of colors used
legend = plt.legend(bars[:len(colors)], legend_labels, title='Categories', loc='upper right')
plt.setp(legend.get_title(), fontsize=12)

plt.show()

As we can see, the clothes category is the most popular among consumers. Let's explore who spends the most money in this category among the top five statuses.
top_locations = df['Location'].value_counts().head(5).index

# Define different colors for bars
colors = ['#98FB98', '#FFE5CC', '#FFCCFF', '#CCE5FF', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Create a subplot grid for each location
fig, axes = plt.subplots(5, 1, figsize=(10, 15))

# Iterate through the top locations and create category distribution plots with different colors
for i, location in enumerate(top_locations):
    location_data = df[df['Location'] == location]
    
    # Count the most common product categories in this location
    category_counts = location_data['Category'].value_counts().head(10)
    
    # Create a bar plot for the category distribution with different colors
    ax = axes[i]
    category_counts.plot(kind='bar', ax=ax, color=colors)
    ax.set_title(f"Categories in {location}")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.set_xticklabels(category_counts.index, rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust subplot layout for a clean appearance
plt.tight_layout()

# Display the visualizations
plt.show()

As we can see, Montana, California, Idaho, Illinois, and Alabama are the top five states in terms of clothing spending.
age_groups = [15, 25, 35, 45, 55, 65]

# Create subplots for each age group
fig, ax = plt.subplots(figsize=(14, 6))

# Create a colormap for age groups
colors = plt.cm.viridis(np.linspace(0, 1, len(age_groups)))

# Initialize a dictionary to store category counts for each age group
category_counts_by_age = {age: [] for age in age_groups}

# Calculate category counts for each age group
for age in age_groups:
    age_group_data = df[(df['Age'] >= age) & (df['Age'] < age + 10)]
    category_counts = age_group_data['Category'].value_counts()
    category_counts_by_age[age] = category_counts

# Create the bar chart
width = 0.15
x = np.arange(len(category_counts_by_age[age_groups[0]].index))

for i, age in enumerate(age_groups):
    category_counts = category_counts_by_age[age]
    ax.bar(x + i * width, category_counts, width=width, label=f'{age}-{age+10}', color=colors[i])

ax.set_xlabel('Product Categories')
ax.set_ylabel('Count')
ax.set_title('Category Distribution by Age Groups')
ax.set_xticks(x + width * (len(age_groups) - 1) / 2)
ax.set_xticklabels(category_counts_by_age[age_groups[0]].index, rotation=45)
ax.legend(title='Age Group')

plt.tight_layout()
plt.show()

As we can see Clothing is the most popular category among the all age groups. Accessories equally famous in all age groups except 15-25 and 65-75 age groups. However, we have seen that in Footwear category is most famous in 45-55 age Groupe. The last one Outerwear is almost equally famous in all age groups.
5. Inights
Inights
Customers tend to make more purchases during winter and autumn compared to spring and summer.
The Outerwear category has slightly lower purchase amounts compared to other categories, indicating potential areas for improvement.
Males account for 67% of the total spending, while females contribute 32% to purchases.
XL size exhibits lower purchases compared to other sizes like Large, Small, and Medium.
The usage of promo codes doesn't seem to have a significant impact on purchase behavior.
Montana stands out with a remarkable number of customers, indicating a thriving consumer market.
Clothing is the most popular product category across all consumer demographics.
Accessories are equally popular across age groups, except for those aged 15-25 and 65-75.
Footwear is particularly popular among the 45-55 age group.
Outerwear enjoys consistent popularity across all age groups.
6. Conclusion
Conclusion
The analysis of customer behavior and purchase data has revealed several valuable insights. Seasonal variations, product categories, gender, size, and promo code usage all play a role in shaping customer purchase decisions. The data also indicates that Montana has a strong consumer market, and clothing is the preferred product category across age groups. These findings can inform marketing strategies, product offerings, and promotions to better target and serve different customer segments.

Request for Feedback
Thank you for taking the time to review my notebook! I'd greatly appreciate your feedback on the following aspects:

Data preprocessing: Did I miss any critical steps?
Corretct Method: Is chosen analytical approch is appropriate for this problem?
Clarity: Is my explanation clear and easy to follow? Your insights will help me improve this analysis. Feel free to comment directly on the notebook or reach out to me. Thank you!
