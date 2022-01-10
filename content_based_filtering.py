import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


#read the Netflix dataset
dataset = pd.read_csv("Netflix Dataset Latest 2021.csv")
dataset.drop(["Unnamed: 29","Unnamed: 30"],axis = 1,inplace=True) #drop irrelavant columns 
dataset["ReleaseYear"] = pd.DatetimeIndex(dataset['Release Date']).year
dataset.rename(columns = {"View Rating":"MaturityRating","Country Availability":"Country","IMDb Score":"IMDbscore"}, inplace = True) #change columns names

#Data cleaning for the Maturity Rtaing column 
def rating(X):
    X.replace("Not Rated","Unrated",inplace = True)
    X.replace("NOT RATED","Unrated",inplace = True)
    X.replace(np.nan,"Unrated",inplace = True)
rating(dataset["MaturityRating"])

#select columns required for the recommendation system
cols = ["Title","Genre", "Languages","MaturityRating","Country","Tags","IMDbscore","ReleaseYear","Summary","Actors"]
df = dataset[cols].copy()
df.dropna(inplace = True)

#split into lists for columns "language", "country", "maturity rating", "tags" . 
cols_list = ["Languages","Genre","Country","MaturityRating","Tags","Actors"]
for i in cols_list:
    df[i] = df[i].apply(lambda x: x.lower().split(",") if len(x.split()) > 0 else x)
df["Title"] = df["Title"].str.lower()
df["Summary"] = df["Summary"].str.lower()

#Clean Tags and split it into a list
def tags(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
df["Tags"] = df["Tags"].apply(tags)

#filter the dataset based on user input
def filter(X,language,country,genre,score,year):
    X_new = pd.DataFrame()
    X = X.loc[(X['IMDbscore'] >= score)]
    X = X.loc[(X['ReleaseYear'] >= year)]
    for i in range(len(X)):
        if language in X.iloc[i]["Languages"]:
            if country in X.iloc[i]["Country"]:
                if genre in X.iloc[i]["Genre"]:
                    X_new = X_new.append(X.iloc[i])
    return(X_new)

#take input from user 
language = input("In which language do you want you want to watch your movie? Ex: English").lower()
country = input("In which country do you want you want to watch your movie? Ex: India").lower()
genre = input("In which Genre are you interested at? Ex: Drama, Romantic").lower()
IMDb_score = int(input("Enter your preferred IMDb rating. Ex: 5,6,7"))
year = float(input("Enter release date of movie. Ex: 2006"))

#filter the dataset and display
data = filter(df,language,country,genre,IMDb_score,year)
data = data.reset_index(drop = True)
display(data["Title"])

#choose a title from the dataset
title = input("Provide a movie title from the above list").lower()

#modify the filtered dataset to be fed into the recommendation system

#df_new = dataset.drop(["Country","Genre","IMDbscore","Languages","Release Date"],axis = 1) #already filtered
data.drop(["IMDbscore","ReleaseYear"],axis = 1,inplace = True)
#df_new["merged"] = df_new["MaturityRating"] + df_new["Tags"]
data["merged"] = data["MaturityRating"] + data["Tags"] + data["Country"] + data["Genre"] + data["Languages"] + data["Actors"]
data["Summary"]= data["Summary"].apply(lambda x: x.split(" "))
data["body"] = data["Summary"]+data["merged"]
data.drop(["MaturityRating","Summary","Tags","merged","Country","Genre","Languages","Actors"],axis = 1,inplace = True )
data["body"] = data['body'].apply(lambda x: ', '.join(map(str, x)))

indices = pd.Series(data.index, index=data['Title']).drop_duplicates()  #get the indices of the movie title. 

#create the TF-idf matrix
tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(data['body'])

#find cosine similarity 
similarity = linear_kernel(matrix,matrix)

def recommend(Title,similarity):
    idx = indices[Title]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores,key=lambda x: x[1],reverse=True)
    high_sim = scores[1:11]
    movie_indices = [i[0] for i in high_sim]
    return data['Title'].iloc[movie_indices]

recommendations = recommend(title,similarity)
print(pd.DataFrame(recommendations.apply(lambda x: x.capitalize())))