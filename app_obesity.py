from pyspark.sql import SparkSession
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from classes_app_obesity import DroppingColumns, StdScaler, OHEnconder
import joblib
from joblib import load

df_obesity = pd.read_csv('https://raw.githubusercontent.com/RamonMMBarbosa/obesity-predictor-fiap/refs/heads/main/tbl_obesity.csv', sep=';', encoding='utf-8')

## PIPELINE DE DADOS

def RunPipeline(df):

    pipeline = Pipeline([
        ('drop_columns', DroppingColumns())
        ,('std_scaler', StdScaler())
        ,('onehot_encoding', OHEnconder())
    ])

    df_encoded = pipeline.fit_transform(df)

    x_test, y_test = df_encoded.drop('obesity', axis=1), df_encoded['obesity']

    return x_test, y_test

## TÍTULO

st.write('# Previsão de obesidade baseado em seu estilo de vida')

## FORMULÁRIO

with st.form(key='form_personal_info'):

    st.write('## **Informações pessoais**')

    input_gender = st.radio('Gênero', ['Homem', 'Mulher'])
    input_gender_dict = {'Homem': 'Male', 'Mulher': 'Female'}
    input_gender = input_gender_dict.get(input_gender)

    input_age = st.number_input('Idade', min_value=0, value=0)

    input_height = st.number_input('Altura (m)', value=0.0, min_value=0.0, max_value=2.5)

    input_weight = st.number_input('Peso (kg)', value=0.0, min_value=0.0, max_value=200.0)

    input_family_history = st.radio('Histórico familiar de obesidade?', ['Sim', 'Não'])
    input_family_history_dict = {'Sim': 1, 'Não': 0}    
    input_family_history = input_family_history_dict.get(input_family_history)

    st.write('## Informações sobre seu estilo de vida')

    ## SMOKE

    input_smoke = st.radio('Você fuma?', ['Sim', 'Não'], key='input_smoke')
    input_smoke_dict = {'Sim': 1, 'Não': 0}    
    input_smoke = input_smoke_dict.get(input_smoke)

    ## TIME_UNDER_ELETRONIC

    input_time_under_eletronic = st.selectbox('Qual frequência você costuma a usar aparelhos eletrônicos diariamente?'
                                           ,['Nunca / Pouco', 'As vezes / Frequentemente', 'A todo momento']
                                           ,key='input_time_under_eletronic'
                                        )
    
    input_time_under_eletronic_dict = {
                                        'Nunca / Pouco': 0
                                        ,'As vezes / Frequentemente': 1
                                        ,'A todo momento': 2
                                        }   
    input_time_under_eletronic = input_time_under_eletronic_dict.get(input_time_under_eletronic)

    ## USE_MOTORIZED_VEHICLE

    input_motorized_vehicle = st.radio('Você utiliza veículos motorizados durante o seu dia a dia (particular ou público)?', ['Sim', 'Não'], key='input_motorized_vehicle')
    input_motorized_vehicle_dict = {'Sim': 1, 'Não': 0}    
    input_motorized_vehicle = input_motorized_vehicle_dict.get(input_motorized_vehicle)

    ## EXERCISE_REGULARLY

    input_exercise_regularly = st.radio('Você faz exercícios diariamente?', ['Sim', 'Não'], key='input_exercise_regularly')
    input_exercise_regularly_dict = {'Sim': 1, 'Não': 0}    
    input_exercise_regularly = input_exercise_regularly_dict.get(input_exercise_regularly)

    ## EXERCISE_FREQUENCY

    if input_exercise_regularly == 1:

            input_exercise_frequency = st.selectbox(
                    'Caso afirmativo na pergunta anterior, quantas vezes na semana?'
                    ,['Nenhuma', 'De 1 a 2 vezes', 'De 3 a 4 vezes', '5 ou mais vezes']
                    ,key='input_exercise_frequency'
            )
            input_exercise_frequency_dict = {
                    'Nenhuma': 0
                    ,'De 1 a 2 vezes': 1
                    ,'De 3 a 4 vezes': 2
                    ,'5 ou mais vezes': 3
            }
            input_exercise_frequency = input_exercise_frequency_dict.get(input_exercise_frequency)

    else:
            
            input_exercise_frequency = st.selectbox(
                    'Caso afirmativo na pergunta anterior, quantas vezes na semana?'
                    ,['Nenhuma']
                    ,disabled=True
                    ,key='input_exercise_frequency'
            )
            input_exercise_frequency = 0

    ## DRINK

    input_drink = st.radio('Você consome bebidas alcoólicas?', ['Sim', 'Não'], key='input_drink')
    input_drink_dict = {'Sim': 1, 'Não': 0}    
    input_drink = input_drink_dict.get(input_drink)

    ## DRINK_FREQUENCY

    if input_drink == 1:

            input_drink_frequency = st.selectbox(
                    'Caso afirmativo na pergunta anterior, com qual frequência?'
                    ,['Socialmente', 'Frequentemente', 'Todo dia']
                    ,key='input_drink_frequency'
            )
            input_drink_frequency_dict = {
                    'Socialmente': 'Sometimes'
                    ,'Frequentemente': 'Frequently'
                    ,'Todo dia': 'Always'
            }
            input_drink_frequency = input_drink_frequency_dict.get(input_drink_frequency)

    else:
            
            input_drink_frequency = st.selectbox(
                    'Caso afirmativo na pergunta anterior, quantas vezes na semana?'
                    ,['Nenhuma']
                    ,disabled=True
                    ,key='input_drink_frequency'
            )
            input_drink_frequency = 'no'

    ## SUBTÍTULO 2

    st.write('## Informações sobre sua alimentação')

    ## HOW_MANY_MEALS_PER_DAY

    input_how_many_meals = st.selectbox(
            'Quantas refeições, incluindo lanches, você faz no dia?'
            ,['De 1 a 2 refeições', 'De 3 a 4 refeições', '5 refeições', 'Mais de 5 refeições']
            ,key='input_how_many_meals'
    )

    input_how_many_meals_dict = {
            'De 1 a 2 refeições': 1
            ,'De 3 a 4 refeições': 2
            ,'5 refeições': 3
            ,'Mais de 5 refeições': 4
    }

    input_how_many_meals = input_how_many_meals_dict.get(input_how_many_meals)

    ## EAT_VEGETABLES

    input_eat_vegetables = st.selectbox(
            'Você come vegetais nas suas refeições?'
            ,['Não como ou como raramente', 'Com frequência', 'Em todas as refeições']
            ,key='input_eat_vegetables'
    )

    input_eat_vegetables_dict = {
            'Não como ou como raramente': 1
            ,'Com frequência': 2
            ,'Em todas as refeições': 3
    }

    input_eat_vegetables = input_eat_vegetables_dict.get(input_eat_vegetables)

    ## EAT_BETWEEN_MEALS

    input_eat_between_meals = st.selectbox(
            'Você come lanches entre suas refeições principais?'
            ,['Não', 'As vezes', 'Com frequência', 'Sempre']
            ,key='input_eat_between_meals'
    )

    input_eat_between_meals_dict = {
            'Não': 'no'
            ,'As vezes': 'Sometimes'
            ,'Com frequência': 'Frequently'
            ,'Sempre': 'Always'
    }

    input_eat_between_meals = input_eat_between_meals_dict.get(input_eat_between_meals)

    ## EAT_CALORICAL_FOOD

    input_eat_calorical_food = st.radio(
            'Sua dieta é composta de alimentos calóricos?'
            ,['Sim', 'Não']
            ,key='input_eat_calorical_food'
    )

    input_eat_calorical_food_dict = {
            'Sim': 1
            ,'Não': 0
    }

    input_eat_calorical_food = input_eat_calorical_food_dict.get(input_eat_calorical_food)

    ## TRACK_CALORIES_EATEN

    input_track_calories_eaten = st.radio(
            'Você acompanha a sua ingestão de calorias diariamente?'
            ,['Sim', 'Não']
            ,key='input_track_calories_eaten'
    )

    input_track_calories_eaten_dict = {
            'Sim': 1
            ,'Não': 0
    }

    input_track_calories_eaten = input_track_calories_eaten_dict.get(input_track_calories_eaten)

    ## WATER_INGESTION_PER_DAY

    input_water_ingestion = st.slider(
            'Quantos litros de água você bebe diariamente?'
            ,min_value=0.0
            ,max_value=10.0
            ,step=0.5
            ,format='%.1f'
            ,key='input_water_ingestion'
    )

    submit_personal_form = st.form_submit_button(label='Enviar respostas')

if submit_personal_form:

    if input_height == 0 or input_weight == 0:
            input_imc = 0
    else:
            input_imc = input_weight / (input_height**2)

    ## SK_GENDER

    input_sk_gender = df_obesity.loc[
    df_obesity['gender'] == input_gender,
    'sk_gender'
    ].values[0]

    ## SK_EAT_BETWEEN_MEALS

    input_sk_eat_between_meals = df_obesity.loc[
    df_obesity['eat_between_meals'] == input_eat_between_meals,
    'sk_eat_between_meals'
    ].values[0]

    ## SK_DRINK_FREQUENCY

    input_sk_drink_frequency = df_obesity.loc[
    df_obesity['drink_frequency'] == input_drink_frequency,
    'sk_drink_frequency'
    ].values[0]

    answers = [
            input_sk_gender             #sk_gender
            ,input_gender               #gender
            ,input_age                  #age
            ,input_height               #height
            ,input_weight               #weight
            ,input_imc                  #imc
            ,input_smoke                #smoke
            ,input_family_history       #family_history
            ,input_time_under_eletronic #time_under_electronic
            ,input_motorized_vehicle    #use_motorized_vehicle
            ,input_how_many_meals       #how_many_meals_per_day
            ,input_eat_calorical_food   #eat_calorical_food
            ,input_eat_vegetables       #eat_vegetables
            ,input_sk_eat_between_meals #sk_eat_between_meals
            ,input_eat_between_meals    #eat_between_meals
            ,input_water_ingestion      #water_ingestion_per_day
            ,input_track_calories_eaten #track_calories_eaten
            ,input_exercise_regularly   #exercises_regularly
            ,input_exercise_frequency   #exercise_frequency
            ,input_sk_drink_frequency   #sk_drink_frequency
            ,input_drink                #drink
            ,input_drink_frequency      #drink_frequency
            ,0                          #obesity
    ]

    answers_columns = [
            'sk_gender'
            ,'gender'
            ,'age'
            ,'height'
            ,'weight'
            ,'imc'
            ,'smoke'
            ,'family_history'
            ,'time_under_eletronics'
            ,'use_motorized_vehicle'
            ,'how_many_meals_per_day'
            ,'eat_calorical_food'
            ,'eat_vegetables'
            ,'sk_eat_between_meals'
            ,'eat_between_meals'
            ,'water_ingestion_per_day'
            ,'track_calories_eaten'
            ,'exercises_regularly'
            ,'exercise_frequency'
            ,'sk_drink_frequency'
            ,'drink'
            ,'drink_frequency'
            ,'obesity'
    ]

    df_answers = pd.DataFrame([answers], columns=answers_columns)

    df_test = pd.concat([df_obesity, df_answers], ignore_index=True)

    df_test = df_test[df_test['drink_frequency'] != 'Always']

    x_test, y_test = RunPipeline(df_test)

    x_test = x_test.iloc[[-1]]

    model = joblib.load('ml_model_obesity.joblib')

    prediction = model.predict(x_test)

    if prediction[0] == 0:
        st.success('✅ Parabéns! Caso mantenha seu estilo de vida, a probabilidade de você ser obeso é muito baixa!')
        st.balloons()
    else:
        st.error('❌ Caso continue a seguir por este estilo de vida, a probabilidade de você tornar-se obeso futuramente é alta!\n' \
        'É aconselhável que procure orientação profissional para readequar sua alimentação e estilo de vida!')