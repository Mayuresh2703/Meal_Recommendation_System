import json
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr

# Load CSS from a file
with open('style1.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def get(path: str):
    with open(path, "r") as p:
        return json.load(p)

path = get("./ani.json")


# Use the full_width property to make the Lottie animation occupy the entire horizontal space
st_lottie(path, width=None)

# Use CSS to make the Lottie animation occupy the entire vertical space
st.markdown("""
    <style>
        div.stLottie {
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, on_bad_lines='skip', delimiter=',')
            # Preprocessing for content-based filtering
            df['calories_from_protein'] = df['protien'] * 4
            df['calories_from_fat'] = df['totalfat'] * 9
            df['calories_from_carbs'] = (df['carbs'] - df['sugar'] - df['addedsugar']) * 4
            df['calories'] = df['calories'].astype(float)
            df['protien'] = df['protien'].astype(float)
            df['totalfat'] = df['totalfat'].astype(float)
            df['carbs'] = df['carbs'].astype(float)
            df['sugar'] = df['sugar'].astype(float)
            df['addedsugar'] = df['addedsugar'].astype(float)
            df['total_calories'] = df['calories']
            df['protien_percentage'] = (df['calories_from_protein'] / df['total_calories']) * 100
            df['fat_percentage'] = (df['calories_from_fat'] / df['total_calories']) * 100
            df['carbs_percentage'] = (df['calories_from_carbs'] / df['total_calories']) * 100

            # Function to create meal feature vector
            def create_meal_feature_vector(row):
                c = row['calories']
                p = row['protien']
                t = row['totalfat']
                ca = row['carbs']
                meal_feature_vector = [c, p, t, ca]
                return meal_feature_vector
              
            # Preparing data for content-based filtering
            meal_feature_vectors = df.apply(create_meal_feature_vector, axis=1)
            meal_feature_matrix = np.array(meal_feature_vectors.tolist())

            
            # Your preprocessing code goes here
            return df
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Please upload a CSV file.")
def content_based_filtering(user_preference_vector, meal_feature_matrix, num_recommendations=10, k=5):
  # Handling potential zero division error
  if sum(user_preference_vector) == 0:
    return [], []
  # Cosine similarity
  similarity_scores = cosine_similarity([user_preference_vector], meal_feature_matrix)
  sorted_indices = np.argsort(similarity_scores)[0][::-1]
  top_meal_indices_cosine = sorted_indices[:num_recommendations]
  # Pearson correlation
  pearson_scores = []
  for feature_vector in meal_feature_matrix:
    if np.std(feature_vector) == 0 or np.std(user_preference_vector) == 0:
      pearson_scores.append(0)  # Handle constant input array
    else:
      pearson_scores.append(pearsonr(user_preference_vector, feature_vector)[0])
      sorted_indices_pearson = sorted(range(len(pearson_scores)), key=lambda i: pearson_scores[i], reverse=True)
      top_indices_pearson = sorted_indices_pearson[:k]

    # Get top recommendations from both methods
    top_meal_indices_pearson = [df.index[i] for i in top_indices_pearson]

    # Return recommendations
  return top_meal_indices_cosine, top_meal_indices_pearson
  
def collaborative_filtering(target_rating_value, df, k=10):
  ratings = df['Ratings'].values.reshape(-1, 1)
  knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
  knn_model.fit(ratings)
  target_rating = [[target_rating_value]]
  distances, indices = knn_model.kneighbors(target_rating)
  nearest_neighbors = df.iloc[indices[0]]
  return nearest_neighbors[['item', 'Ratings']]      

def main():
    st.title('MEAL RECOMMENDATION SYSTEM')
    uploaded_file = st.file_uploader("Please Load your dataset", type="csv")
    
    # Check if the button is clicked
    if st.button("Let's find the best for you!!"):
        # Switch to the meal recommender page
        meal_recommender(uploaded_file)
        return
    
    # Display the main page
    st_lottie(path, width=None)
    st.markdown("""
        <style>
            div.stLottie {
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
        </style>
    """, unsafe_allow_html=True)

def meal_recommender(uploaded_file):
    st.header("Please Load your dataset")
    df = load_data(uploaded_file)
    
    if df is not None:
        # Your code for meal recommender goes here
        st.subheader("Meal Recommender")
        option = st.sidebar.selectbox("Choose an option", ["Health Conscious", "Tasty Foods"])

        if option == "Health Conscious":
            st.subheader("Healthy Meals")
            calorie_requirement = st.number_input("Enter your daily calorie requirement", min_value=0.0)
        
            if calorie_requirement == 0:
                st.warning("Your daily calorie requirement please")
                return

            protein_ratio = st.number_input("Percentage of calories from protein", min_value=0.0, max_value=100.0)
            fat_ratio = st.number_input("Percentage of calories from fat", min_value=0.0, max_value=100.0)
            carbohydrate_ratio = st.number_input("Percentage of calories from carbohydrates", min_value=0.0, max_value=100.0)
            user_preference_vector = [calorie_requirement, protein_ratio, fat_ratio, carbohydrate_ratio]
            user_preference_vector = [value / sum(user_preference_vector) for value in user_preference_vector]

            # Get recommendations
            top_meal_indices_cosine, top_meal_indices_pearson = content_based_filtering(user_preference_vector, meal_feature_matrix)
        
            # Convert indices to item names
            recommended_meals_cosine = [df.iloc[i]['item'] for i in top_meal_indices_cosine]
            recommended_meals_pearson = [df.iloc[i]['item'] for i in top_meal_indices_pearson]

            # Find common recommendations
            common_recommendations = list(set(recommended_meals_cosine) & set(recommended_meals_pearson))

            if len(common_recommendations)==0:
                st.write("No common recommendations found.")
            else:
                st.subheader("Top Recommended Meals (Health Conscious):")
                for i, meal in enumerate(common_recommendations):
                    st.write(f"{i + 1}. {meal}")

        elif option == "Tasty Foods":
            st.subheader("Delicious Meals")
            target_rating_value = st.number_input("Enter the rating of food you desire", min_value=1.0, max_value=20.0)
            recommended_meals = collaborative_filtering(target_rating_value, df)
            st.subheader("Top Recommended Meals (Tasty Foods):")
            st.write(recommended_meals)

if __name__ == "__main__":
    main()
