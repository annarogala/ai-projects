# Movie recommendation engine

The program is a movie recommendation engine.  
It uses the Surprise library to train a recommendation model and provide movie recommendations for a specific user.  
The program reads from an Excel file, handles missing values, and transforms the data into a format where each row represents a user, a movie, and a rating.  
It uses the Pearson and Cosine metrics to calculate the distance between users.  
The output is top 5 movie recommendations and 5 movies not recommended for a specific user.

## How to set up:
Install the packages from the requirements.txt with the following command `pip3 install -r requirements.txt`

## How to run:
Run the program with the following command `python3 movie_recommendation_engine.py`  
You will be asked to provide a user for whom the recommendations are to be generated.  
Please type in user name exactly as it is in the Excel file.  
The program will print top 5 movie recommendations and 5 movies not recommended with both the Pearson and Cosine metrics.  

## Usage example:

**Example 1**:
![Usage video](mre-demo-1.mp4)

**Example 2**:
![Usage video](mre-demo-2.mp4)

**Example 3**:
![Usage video](mre-demo-3.mp4)

**Example 4**:
![Usage video](mre-demo-4.mp4)

## Authors:
Adam Łuszcz s22994  
Anna Rogala s21487

## Sources:
- https://surprise.readthedocs.io/
- https://pandas.pydata.org/docs/
