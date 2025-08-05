import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
from collections import Counter
from ctypes import create_string_buffer
from scipy.optimize import linprog  
import numpy as np
import itertools
import matplotlib.pyplot as plt
import textwrap


st.set_page_config(page_title="Dog Diet Recommendation", layout="centered")

@st.cache_data(show_spinner=False)
def load_data():
    food = pd.read_csv("FINAL_COMBINED.csv")
    disease = pd.read_csv("Disease.csv")
    return food, disease

food_df, disease_df = load_data()

df_standart = pd.read_csv("merge_tab.csv")
proteins=df_standart[df_standart["Type"].isin(["Яйца и Молочные продукты", "Мясо"])]["Ingredient"].tolist()
oils=df_standart[df_standart["Type"].isin([ "Масло и жир"])]["Ingredient"].tolist()
carbonates_cer=df_standart[df_standart["Type"].isin(["Крупы"])]["Ingredient"].tolist()
carbonates_veg=df_standart[df_standart["Type"].isin(["Зелень и специи","Овощи и фрукты"])]["Ingredient"].tolist()
other=df_standart[df_standart["Type"].isin(["Вода, соль и сахар"])]["Ingredient"].tolist()
water=["Вода — Обыкновенный"]
dele = df_standart[df_standart["Standart"].isna()]["Ingredient"].tolist()

stop_words=["Beta-Carotene","With Natural Antioxidant", "Minerals","Digest","Dicalcium Phosphate","L-Carnitine","L-Threonine","Composition:","L-Tryptophan","Chicken Flavor","Manganese Sulfate"
"Hydrolyzed Chicken Flavor", "Monosodium Phosphate","Magnesium Oxide","Powdered Cellulose","Taurine","Mixed Tocopherols For Freshness","Natural Flavor","Potassium Alginate","Sodium Tripolyphosphate",
"Dl-Methionine","Calcium Sulfate","Guar Gum","Betaine","Glyceryl Monostearate","Calcium Chloride","Calcium Lactate","Calcium Gluconate","Natural Flavors","Choline Chloride","Calcium Iodate",
"Dextrose","Zinc Oxide","Copper Sulfate","Ferrous Sulfate","Niacin Supplement","Thiamine Mononitrate","Calcium Pantothenate","Riboflavin Supplement","Biotin",'Pyridoxine Hydrochloride',
"Folic Acid","Disodium Phosphate","Potassium Chloride","Chondroitin Sulfate","Copper Proteinate","Potassium Iodide)","Sodium Pyrophosphate","Sodium Hexametaphosphate","Carrageenan",
"Manganous Oxide","Sodium Selenite","Lipoic Acid","Calcium Carbonate","Vitamin A Supplement","Manganese Sulfate","Derivatives Of Vegetable Origin","Cellulose","Potassium Citrate","Glycerin","Vegetable Protein Extracts",
"Manganese Sulfate","Caramel Color","Citric Acid For Freshness","Brewers Dried Yeast","Soybean Mill Run","Glucosamine Hydrochloride","Vitamin A Supplement","Pork Plasma","Pork Gelatin"]

if "step" not in st.session_state:
    st.session_state.step = 0



def classify_breed_size(row):
    w = (row["min_weight"] + row["max_weight"]) / 2
    if w <= 10:
        return "Small Breed"
    elif w <= 25:
        return "Medium Breed"
    else:
        return "Large Breed"

@st.cache_data(show_spinner=False)
def preprocess_disease(df):
    df = df.copy()
    df["breed_size_category"] = df.apply(classify_breed_size, axis=1)
    return df

disease_df = preprocess_disease(disease_df)

@st.cache_data(show_spinner=False)
def preprocess_food(df):
    df = df.copy()
    nutrients = [
        "protein", "fat", "carbohydrate (nfe)", "crude fibre", "calcium",
        "phospohorus", "potassium", "sodium", "magnesium", "vitamin e",
        "vitamin c", "omega-3-fatty acids", "omega-6-fatty acids",
    ]
    for col in nutrients:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "")
            .str.replace("IU/kg", "")
            .str.extract(r"([\d\.]+)")
            .astype(float)
            .fillna(0.0)
        )

    df["combined_text"] = (
        df["ingredients"].fillna("")
        .str.cat(df["key benefits"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["product title"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["product description"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["helpful tips"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["need/preference"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["alternate product recommendation"].fillna(""), sep=" ", na_rep="")
    )
    return df

food_df = preprocess_food(food_df)

# -----------------------------------
# 4) TEXT VECTORIZATION & SVD
# -----------------------------------

@st.cache_resource(show_spinner=False)
def build_text_pipeline(corpus, n_components=100):
    vect = TfidfVectorizer(stop_words="english", max_features=5000)
    X_tfidf = vect.fit_transform(corpus)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)

    return vect, svd, X_reduced

vectorizer, svd, X_text_reduced = build_text_pipeline(food_df["combined_text"], n_components=100)

# -----------------------------------
# 5) CATEGORICAL ENCODING
# -----------------------------------

@st.cache_resource(show_spinner=False)
def build_categorical_encoder(df):
    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    cats = df[["breed size", "lifestage"]].fillna("Unknown")
    enc.fit(cats)
    return enc, enc.transform(cats)

encoder, X_categorical = build_categorical_encoder(food_df)

# -----------------------------------
# 6) COMBINE FEATURES INTO SPARSE MATRIX
# -----------------------------------

@st.cache_resource(show_spinner=False)
def combine_features(text_reduced, _cat_matrix):
    # Turn dense text_reduced into sparse form
    X_sparse_text = csr_matrix(text_reduced)
    return hstack([X_sparse_text, _cat_matrix])

X_combined = combine_features(X_text_reduced, X_categorical)

# -----------------------------------
# 7) TRAIN RIDGE REGRESSORS FOR NUTRIENTS
# -----------------------------------

@st.cache_resource(show_spinner=False)
def train_nutrient_models(food, _X):
    nutrient_models = {}
    scalers = {}

    nutrients = [
        "protein", "fat", "carbohydrate (nfe)", "crude fibre", "calcium",
        "phospohorus", "potassium", "sodium", "magnesium", "vitamin e",
        "vitamin c", "omega-3-fatty acids", "omega-6-fatty acids",
    ]
    to_scale = {
        "sodium",
        "omega-3-fatty acids",
        "omega-6-fatty acids",
        "calcium",
        "phospohorus",
        "potassium",
        "magnesium",
    }

    for nutrient in nutrients:
        y = food[nutrient].fillna(food[nutrient].median()).values.reshape(-1, 1)
        if nutrient in to_scale:
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y).ravel()
        else:
            scaler = None
            y_scaled = y.ravel()

        X_train, _, y_train, _ = train_test_split(_X, y_scaled, test_size=0.2, random_state=42)

        base = Ridge()
        search = GridSearchCV(
            base,
            param_grid={"alpha": [0.1, 1.0]},
            scoring="r2",
            cv=2,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        nutrient_models[nutrient] = search.best_estimator_
        scalers[nutrient] = scaler

    return nutrient_models, scalers

# **This line must run at import-time** so ridge_models is defined before you use it below:
ridge_models, scalers = train_nutrient_models(food_df, X_combined)

# -----------------------------------
# 8) TRAIN RIDGE CLASSIFIERS FOR INGREDIENT PRESENCE
# -----------------------------------

@st.cache_resource(show_spinner=False)
def train_ingredient_models(food, _X):
    all_ings = []
    for txt in food["ingredients"].dropna():
        tokens = [i.strip().lower() for i in txt.split(",")]
        all_ings.extend(tokens)

    counts = Counter(all_ings)
    frequent = [ing for ing, cnt in counts.items() if cnt >= 5]

    targets = {}
    low = food["ingredients"].fillna("").str.lower()
    for ing in frequent:
        targets[ing] = low.apply(lambda s: int(ing in s)).values

    ing_models = {}
    for ing, y in targets.items():
        clf = RidgeClassifier()
        clf.fit(_X, y)
        ing_models[ing] = clf

    return ing_models, frequent

# **This line must run at import-time** so ingredient_models is defined before you use it below:
ingredient_models, frequent_ingredients = train_ingredient_models(food_df, X_combined)

# -----------------------------------
# 9) DISORDER KEYWORDS DICTIONARY
# -----------------------------------

disorder_keywords = {
    "Inherited musculoskeletal disorders": "joint mobility glucosamine arthritis cartilage flexibility",
    "Inherited gastrointestinal disorders": "digest stomach bowel sensitive diarrhea gut ibs",
    "Inherited endocrine disorders": "thyroid metabolism weight diabetes insulin hormone glucose",
    "Inherited eye disorders": "vision eye retina cataract antioxidant sight ocular",
    "Inherited nervous system disorders": "brain seizure cognitive nerve neuro neurological cognition",
    "Inherited cardiovascular disorders": "heart cardiac circulation omega-3 blood pressure vascular",
    "Inherited skin disorders": "skin allergy itch coat omega-6 dermatitis eczema flaky",
    "Inherited immune disorders": "immune defense resistance inflammatory autoimmune",
    "Inherited urinary and reproductive disorders": "urinary bladder kidney renal urine reproductive",
    "Inherited respiratory disorders": "breath respiratory airway lung cough breathing nasal",
    "Inherited blood disorders": "anemia blood iron hemoglobin platelets clotting hemophilia",
}

# -----------------------------------
# 10) STREAMLIT UI LAYOUT
# -----------------------------------
ingredients_finish=[]
st.sidebar.title("🐶 Smart Dog Diet Advisor")
st.sidebar.write("Select breed + disorder → get personalized food suggestions")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=80)

st.header("Dog Diet Recommendation")

breed_list = sorted(disease_df["Breed"].unique())
user_breed = st.selectbox("Select dog breed:", breed_list)

if user_breed:
    info = disease_df[disease_df["Breed"] == user_breed]
    if not info.empty:
        breed_size = info["breed_size_category"].values[0]
        disorders = info["Disease"].unique().tolist()
        selected_disorder = st.selectbox("Select disorder:", disorders)
        disorder_type = info[info["Disease"] == selected_disorder]["Disorder"].values[0]

        if st.session_state.step == 0:
         if st.button("Generate Recommendation"):
            st.session_state.step = 1
            st.rerun()  # чтобы обновить UI
            # 10.1) Build query vector
            keywords = disorder_keywords.get(disorder_type, selected_disorder).lower()
            kw_tfidf = vectorizer.transform([keywords])
            kw_reduced = svd.transform(kw_tfidf)

            # One-hot for (breed_size, "Adult")
            cat_vec = encoder.transform([[breed_size, "Adult"]])
            kw_combined = hstack([csr_matrix(kw_reduced), cat_vec])

            # 10.2) Predict nutrients
            nutrient_preds = {}
            for nut, model in ridge_models.items():
                pred = model.predict(kw_combined)[0]
                sc = scalers.get(nut)
                if sc:
                    pred = sc.inverse_transform([[pred]])[0][0]
                nutrient_preds[nut] = round(pred, 2)

            # 10.3) Rank ingredients
            ing_scores = {
                ing: clf.decision_function(kw_combined)[0]
                for ing, clf in ingredient_models.items()
            }
            top_ings = sorted(ing_scores.items(), key=lambda x: x[1], reverse=True)[:20]

            prot=sorted([i for i in top_ings if i[0].title() in proteins and i[0].title() not in dele], key=lambda x: x[1], reverse=True)[:3]
            prot = [i.title() for i, _ in prot]
            prot=df_standart[df_standart["Ingredient"].isin(prot)]["Standart"].tolist()

            carb_cer=sorted([i for i in top_ings if i[0].title() in carbonates_cer and i[0].title() not in dele], key=lambda x: x[1], reverse=True)[:1]
            carb_cer = [i.title() for i, _ in carb_cer]
            carb_cer=df_standart[df_standart["Ingredient"].isin(carb_cer)]["Standart"].tolist()

            carb_veg=sorted([i for i in top_ings if i[0].title() in carbonates_veg and i[0].title() not in dele], key=lambda x: x[1], reverse=True)[:1]
            carb_veg = [i.title() for i, _ in carb_veg]
            carb_veg=df_standart[df_standart["Ingredient"].isin(carb_veg)]["Standart"].tolist()


            fat=sorted([i for i in top_ings if i[0].title() in oils and i[0].title() not in dele], key=lambda x: x[1], reverse=True)[:1]
            fat = [i.title() for i, _ in fat]
            fat=df_standart[df_standart["Ingredient"].isin(fat)]["Standart"].tolist()

            oth=sorted([i for i in top_ings[:20] if i[0].title() in other and i[0].title() not in dele], key=lambda x: x[1], reverse=True)[:1]
            if len(oth)>0:
              oth = [i.title() for i, _ in oth]
              oth=df_standart[df_standart["Ingredient"].isin(oth)]["Standart"].tolist()
            else:
              oth=[]
            
            ingredients_finish = [i for i in list(set(prot))+list(set(carb_cer+carb_veg+fat))+list(set(oth+water)) if len(i)>0]
                     
            # 10.5) Display
            st.subheader("🌿 Recommended Ingredients")
            st.write(f"Based on disorder: **{disorder_type}**")
            for ing in ingredients_finish:
                st.write("• " + ing)





            if len(ingredients_finish)>0:
               
                      # --- Загрузка данных ---
                      df_ingr_all = pd.read_csv('food_ingrediets.csv')
                      cols_to_divide = ['Влага', 'Белки', 'Углеводы', 'Жиры']



                      for col in cols_to_divide:
                          df_ingr_all[col] = df_ingr_all[col].astype(str).str.replace(',', '.', regex=False)
                          df_ingr_all[col] = pd.to_numeric(df_ingr_all[col], errors='coerce')

                      df_ingr_all[cols_to_divide] = df_ingr_all[cols_to_divide] / 100
                      df_ingr_all['ингредиент и описание'] = df_ingr_all['Ингредиенты'] + ' — ' + df_ingr_all['Описание']


                      proteins=df_ingr_all[df_ingr_all["Категория"].isin(["Яйца и Молочные продукты", "Мясо"])]["ингредиент и описание"].tolist()
                      oils=df_ingr_all[df_ingr_all["Категория"].isin([ "Масло и жир"])]["ингредиент и описание"].tolist()
                      carbonates_cer=df_ingr_all[df_ingr_all["Категория"].isin(["Крупы"])]["ингредиент и описание"].tolist()
                      carbonates_veg=df_ingr_all[df_ingr_all["Категория"].isin(["Зелень и специи","Овощи и фрукты"])]["ингредиент и описание"].tolist()
                      other=df_ingr_all[df_ingr_all["Категория"].isin(["Вода, соль и сахар"])]["ингредиент и описание"].tolist()

                      meat_len=len(set(proteins).intersection(set(ingredients_finish)))



                      if "selected_ingredients" not in st.session_state:
                          # Преобразуем ingredients_finish в set и сохраняем
                          st.session_state.selected_ingredients = set(ingredients_finish)

                      st.title("🍲 Выбор ингредиентов")
                      for category in df_ingr_all['Категория'].dropna().unique():
                          with st.expander(f"{category}"):
                              df_cat = df_ingr_all[df_ingr_all['Категория'] == category]
                              for ingredient in df_cat['Ингредиенты'].dropna().unique():
                                  with st.expander(f"{ingredient}"):
                                      df_ing = df_cat[df_cat['Ингредиенты'] == ingredient]
                                      for desc in df_ing['Описание'].dropna().unique():
                                          label = f"{ingredient} — {desc}"
                                          key = f"{category}_{ingredient}_{desc}"
                                          if st.button(f"{desc}", key=key):
                                              st.session_state.selected_ingredients.add(label)   

                      st.markdown("### ✅ Выбранные ингредиенты:")
                      for i in sorted(st.session_state.selected_ingredients):
                          col1, col2 = st.columns([5, 1])
                          col1.write(i)
                          if col2.button("❌", key=f"remove_{i}"):
                              st.session_state.selected_ingredients.remove(i)
                      else:
                          st.info("Вы пока не выбрали ни одного ингредиента.")

                      # Пример: доступ к выбранным
                      ingredient_names = list(st.session_state.selected_ingredients)
                      food = df_ingr_all.set_index("ингредиент и описание")[cols_to_divide].to_dict(orient='index')


                      # --- Ограничения по количеству каждого ингредиента ---
                      if ingredient_names:
                          st.subheader("Ограничения по количеству ингредиентов (в % от 100 г):")
                          ingr_ranges = []
                          for ingr in ingredient_names:
                              if ingr in proteins:
                                ingr_ranges.append(st.slider(f"{ingr}", 0, 100, value=(int(40 / meat_len), int(60 / meat_len))))

                              elif ingr in oils:
                                ingr_ranges.append(st.slider(f"{ingr}", 0, 100, (1,10)))

                              elif ingr in carbonates_cer:
                                ingr_ranges.append(st.slider(f"{ingr}", 0, 100, (10,35)))

                              elif ingr in carbonates_veg:
                                ingr_ranges.append(st.slider(f"{ingr}", 0, 100, (10,25)))
                              elif "Вода" in ingr:
                                ingr_ranges.append(st.slider(f"{ingr}", 0, 100, (0,30)))
                              elif ingr in other:
                                  ingr_ranges.append(st.slider(f"{ingr}", 0, 100, (1,3)))


                          # --- Ограничения по нутриентам ---
                          st.subheader("Ограничения по нутриентам:")
                          nutr_ranges = {}
                          nutr_ranges['Влага'] = st.slider(f"{'Влага'}", 0, 100, (70, 85))
                          nutr_ranges['Белки'] = st.slider(f"{'Белки'}", 0, 100, (int(float(nutrient_preds["protein"])-2),int(float(nutrient_preds["protein"])+2)))
                          nutr_ranges['Углеводы'] = st.slider(f"{'Углеводы'}", 0, 100, (5,10))
                          nutr_ranges['Жиры'] = st.slider(f"{'Жиры'}", 0, 100, (5,15))

                          # --- Построение задачи LP ---
                          A = [
                              [food[ing][nutr] if val > 0 else -food[ing][nutr]
                              for ing in ingredient_names]
                              for nutr in nutr_ranges
                              for val in (-nutr_ranges[nutr][0]/100, nutr_ranges[nutr][1]/100)
                          ]
                          b = [
                              val / 100 for nutr in nutr_ranges
                              for val in (-nutr_ranges[nutr][0], nutr_ranges[nutr][1])
                          ]

                          A_eq = [[1 for _ in ingredient_names]]
                          b_eq = [1.0]
                          bounds = [(low/100, high/100) for (low, high) in ingr_ranges]

                          # --- Целевая функция ---
                          st.subheader("Что максимизировать?")
                          selected_maximize = st.multiselect(
                              "Выберите нутриенты для максимизации:",
                              cols_to_divide,
                              default=['Влага',"Белки"]
                          )

                          f = [-sum(food[i][nutr] for nutr in selected_maximize) for i in ingredient_names]

                        
                          if st.session_state.step == 1:
                           if st.button("🔍 Рассчитать оптимальный состав"):
                              res = linprog(f, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

                              if res.success:
                                  st.success("✅ Решение найдено!")
                                  result = {name: round(val * 100, 2) for name, val in zip(ingredient_names, res.x)}
                                  st.markdown("### 📦 Состав (в граммах на 100 г):")
                                  for name, value in result.items():
                                      st.write(f"{name}: **{value} г**")

                                  st.markdown("### 💪 Питательная ценность на 100 г:")
                                  nutrients = {
                                      nutr: round(sum(res.x[i] * food[name][nutr] for i, name in enumerate(ingredient_names)) * 100, 2)
                                      for nutr in cols_to_divide
                                  }
                                  for k, v in nutrients.items():
                                      st.write(f"**{k}:** {v} г")
                              else:
                                  st.error("❌ Не удалось найти оптимальное решение. Попробуйте другие параметры.")

                      else:
                          st.info("👈 Пожалуйста, выберите хотя бы один ингредиент.")


    else:
        st.info("No disease info found for this breed.")
else:
    st.info("Please select a breed to continue.")

