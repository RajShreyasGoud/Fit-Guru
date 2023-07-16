import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

df = pd.read_csv("C:/Users/rajsh/Desktop/D 13 Mini project/ObesityDataSet_raw_and_data_sinthetic.csv")
st.title("Fit Guru")
st.header("Eat healthy , Stay Fit , Live Long")
df_y = df[['NObeyesdad']]

df_bool = df[['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']]

df_xstr = df[['CAEC', 'CALC', 'MTRANS']]

df_xnum = df[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']]
le = preprocessing.LabelEncoder()
df_xEn = df_xstr.apply(le.fit_transform)
# OneHotEncoder
enc = preprocessing.OneHotEncoder()
enc.fit(df_bool)
df_xOHE = enc.transform(df_bool).toarray()

array_x = np.concatenate((df_xnum, df_xEn, df_xOHE), axis=1)
df_x = pd.DataFrame(array_x)
df_x.columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'CAEC', 'CALC', 'MTRANS', 'Female',
                'Male', 'Overweight history_No', 'Overweight history_Yes', 'FAVC_No', 'FAVC_Yes', 'SMOKE_No',
                'SMOKE_Yes', 'SCC_No', 'SCC_Yes']

from sklearn.model_selection import train_test_split

dfx_train, dfx_test, dfy_train, dfy_test = train_test_split(df_x, df_y, test_size=0.1)

# Tree classifier training - hyper parameter max depth
tree_clf = DecisionTreeClassifier(max_depth=10)
tree_clf.fit(dfx_train, dfy_train)

# Predictoin results vs original data
Test_predict = pd.DataFrame(tree_clf.predict(dfx_test))

# Plots for results

Val1 = Test_predict.value_counts()
Val2 = dfy_test.value_counts()
Accuracy= accuracy_score(dfy_test, Test_predict)
#st.write(Accuracy)
# Interface to enter new instances
Array_Ans = [None] * 11
Array_Ans[0] = st.number_input('Age:')
Array_Ans[1] = st.number_input('Height in meters:')
Array_Ans[2] = st.number_input('Weight in kg:')

d1={"Never":3, "Sometimes":2, "Always":1}
Array_Ans[3] = d1[st.selectbox('Do you usually eat vegetables in your meals?' , ("Never", "Sometimes", "Always"))]

d2 = {"One to two":3, "Three":2, "More than three":1}
Array_Ans[4] = d2[st.selectbox('How many main meals do you have daily?' , ("One to two", "Three", "More than three"))]

d3={"Less than a liter":3, "One or two liters":2, "More than two liters":1}
Array_Ans[5] = d3[st.selectbox('How much water do you drink daily?' , ("Less than a liter", "One or two liters", "More than two liters"))]

d4 = {"0 to 1 day":3, "1 or 2 days":2, "2 or 4 days":1, "More than 4":0}
Array_Ans[6] = d4[st.selectbox('How often do you exercise?' , ("0 to 1 day", "1 or 2 days", "2 or 4 days", "More than 4"))]

d5 = {"0 to 2 hours":2, "3 to 5 hours":1, "More than 5":0}
Array_Ans[7] = d5[st.selectbox('Hours spent using electronic devices?' , ("0 to 2 hours", "3 to 5 hours", "More than 5"))]

d6 = {"No":3, "Sometimes":2, "Frequently":1, "Always":0}
Array_Ans[8] = d6[st.selectbox('Do you eat between meals?' , ("No", "Sometimes", "Frequently", "Always"))]

d7 = {"Never":3, "Sometimes":2, "Frequently":1, "Always":0}
Array_Ans[9] = d7[st.selectbox('How often do you drink alcohol?' , ("Never", "Sometimes", "Frequently", "Always"))]

d8 = {"Automobile":0, "Motorbike":1, "Bike":2, "Public_Transportation":3, "Walking":4}
Array_Ans[10] = d8[st.selectbox('From the above, choose your transportation method:' ,("Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"))]

Ans_str = [[None] * 5]
Ans_str[0][0] = st.radio('Gender', ['female', 'male'])
Ans_str[0][1] = st.radio('Are there any cases of obesity in your family?', ['yes', 'no'])
Ans_str[0][2] = st.radio('Do you eat high caloric food frequently?', ['yes', 'no'])
Ans_str[0][3] = st.radio('Do you smoke?', ['yes', 'no'])
Ans_str[0][4] = st.radio('Do you monitor your calories?', ['yes', 'no'])

# Perform one-hot encoding for the answer array
Array_OHE = [None] * 10
for i in range(5):
    if i == 0:
        if Ans_str[0][i] == 'female':
            Array_OHE[0] = 1
            Array_OHE[1] = 0
        else:
            Array_OHE[0] = 0
            Array_OHE[1] = 1
    else:
        if Ans_str[0][i] == 'no':
            Array_OHE[i * 2] = 1
            Array_OHE[(i * 2) + 1] = 0
        else:
            Array_OHE[i * 2] = 0
            Array_OHE[(i * 2) + 1] = 1

UserAns = np.concatenate((Array_Ans, Array_OHE))

# User's prediction
st.write("Prediction for new user")
probs = tree_clf.predict_proba([UserAns])
weight_classification = tree_clf.predict([UserAns])[0]

st.write("The user's weight was classified into the following class:", weight_classification)
st.write("WEEKLY DIET PLAN")

cuisine = st.selectbox('Cuisine', ['continental', 'Indian'])

diet_plan = {
    'Obesity_Type_III': {
        'continental': '''
           Day 1
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Grilled chicken salad with whole-wheat bread
                Dinner: Salmon with roasted vegetables
            Day 2
                
                Breakfast: Yogurt with fruit and granola
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Lentil soup with whole-wheat bread
            Day 3
                
                Breakfast: Eggs with whole-wheat toast and avocado
                Lunch: Salad with grilled chicken or tofu
                Dinner: Stir-fry with vegetables and brown rice
            Day 4
                
                Breakfast: Smoothie made with yogurt, fruit, and spinach
                Lunch: Chickpea curry with rice
                Dinner: Tofu stir-fry with vegetables
            Day 5
                
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
                
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables
''',
        'Indian': '''
            Day 1

                Breakfast: Oatmeal with berries and nuts
                Lunch: Chicken tikka masala with brown rice and vegetables
                Dinner: Lentil soup with whole-wheat bread
            Day 2
                
                Breakfast: Yogurt with fruit and granola
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables
            Day 3
                
                Breakfast: Eggs with whole-wheat toast and avocado
                Lunch: Salad with grilled chicken or tofu
                Dinner: Stir-fry with vegetables and brown rice
            Day 4
                
                Breakfast: Smoothie made with yogurt, fruit, and spinach
                Lunch: Chickpea curry with rice
                Dinner: Tofu stir-fry with vegetables
            Day 5
                
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
                
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables
        '''
    },
    'Obesity_Type_I': {
        'continental': '''
            Day 1

                Breakfast: Oatmeal with berries and nuts
                Lunch: Grilled chicken salad with whole-wheat bread
                Dinner: Salmon with roasted vegetables
            Day 2
                
                Breakfast: Yogurt with fruit and granola
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Lentil soup with whole-wheat bread
            Day 3
                
                Breakfast: Eggs with whole-wheat toast and avocado
                Lunch: Salad with grilled chicken or tofu
                Dinner: Stir-fry with vegetables and brown rice
            Day 4
                
                Breakfast: Smoothie made with yogurt, fruit, and spinach
                Lunch: Chickpea curry with rice
                Dinner: Tofu stir-fry with vegetables
            Day 5
                
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
                
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables

        ''',
        'Indian': '''
           Day 1
                Breakfast: Vegetable poha with curd
                Lunch: Dal makhani with roti and salad
                Dinner: Chicken tikka masala with brown rice and vegetables
            Day 2
            
                Breakfast: Idli with sambar and chutney
                Lunch: Vegetable biryani with raita
                Dinner: Fish curry with steamed rice and vegetables
            Day 3
            
                Breakfast: Upma with vegetables
                Lunch: Chicken salad sandwich
                Dinner: Tofu stir-fry with vegetables
            Day 4
            
                Breakfast: Ragi porridge with milk and nuts
                Lunch: Lentil soup with whole-wheat bread
                Dinner: Vegetable pulao with raita
            Day 5
            
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
            
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
            
                Breakfast: Vegetable poha with curd
                Lunch: Dal makhani with roti and salad
                Dinner: Chicken tikka masala with brown rice and vegetables

                    '''
    },
    'Normal_Weight': {
        'continental': '''
            Day 1
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Salad with grilled chicken or tofu
                Dinner: Salmon with roasted vegetables
            Day 2
                
                Breakfast: Yogurt with fruit and granola
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Lentil soup with whole-wheat bread
            Day 3
                
                Breakfast: Eggs with whole-wheat toast and avocado
                Lunch: Salad with grilled chicken or tofu
                Dinner: Stir-fry with vegetables and brown rice
            Day 4
                
                Breakfast: Smoothie made with yogurt, fruit, and spinach
                Lunch: Chickpea curry with rice
                Dinner: Tofu stir-fry with vegetables
            Day 5
                
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
                
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables

        ''',
        'Indian': '''
           Day 1

                Breakfast: Oatmeal with berries and nuts
                Lunch: Chicken tikka masala with brown rice and vegetables
                Dinner: Lentil soup with whole-wheat bread
            Day 2
            
                Breakfast: Yogurt with fruit and granola
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables
            Day 3
            
                Breakfast: Eggs with whole-wheat toast and avocado
                Lunch: Salad with grilled chicken or tofu
                Dinner: Stir-fry with vegetables and brown rice
            Day 4
            
                Breakfast: Smoothie made with yogurt, fruit, and spinach
                Lunch: Chickpea curry with rice
                Dinner: Tofu stir-fry with vegetables
            Day 5
            
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
            
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
            
                Breakfast: Oatmeal with berries and nuts
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables

        '''
    },
    'Obesity_Type_II': {
        'continental': '''
            Day 1:
            
                Breakfast: Oatmeal with berries and nuts
                Lunch: Salad with grilled chicken or tofu
                Dinner: Salmon with roasted vegetables
            Day 2
                
                Breakfast: Yogurt with fruit and granola
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Lentil soup with whole-wheat bread
            Day 3
                
                Breakfast: Eggs with whole-wheat toast and avocado
                Lunch: Salad with grilled chicken or tofu
                Dinner: Stir-fry with vegetables and brown rice
            Day 4
                
                Breakfast: Smoothie made with yogurt, fruit, and spinach
                Lunch: Chickpea curry with rice
                Dinner: Tofu stir-fry with vegetables
            Day 5
                
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
                
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables
        ''',
        'Indian': '''
            Day 1

                Breakfast: Oatmeal with berries and nuts
                Lunch: Chicken tikka masala with brown rice and vegetables
                Dinner: Lentil soup with whole-wheat bread
            Day 2
                
                Breakfast: Yogurt with fruit and granola
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables
            Day 3
                
                Breakfast: Eggs with whole-wheat toast and avocado
                Lunch: Salad with grilled chicken or tofu
                Dinner: Stir-fry with vegetables and brown rice
            Day 4
                
                Breakfast: Smoothie made with yogurt, fruit, and spinach
                Lunch: Chickpea curry with rice
                Dinner: Tofu stir-fry with vegetables
            Day 5
                
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
                
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables

                '''
    },
    'Overweight_Level_II': {
        'continental': '''
            Day 1
            
                Breakfast: Oatmeal with berries and nuts
                Lunch: Salad with grilled chicken or tofu
                Dinner: Salmon with roasted vegetables
            Day 2
            
                Breakfast: Yogurt with fruit and granola
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Lentil soup with whole-wheat bread
            Day 3
            
                Breakfast: Eggs with whole-wheat toast and avocado
                Lunch: Salad with grilled chicken or tofu
                Dinner: Stir-fry with vegetables and brown rice
            Day 4
            
                Breakfast: Smoothie made with yogurt, fruit, and spinach
                Lunch: Chickpea curry with rice
                Dinner: Tofu stir-fry with vegetables
            Day 5
            
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
            
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
            
                Breakfast: Oatmeal with berries and nuts
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables
        ''',
        'Indian': '''
            Day 1

                Breakfast: Oatmeal with berries and nuts
                Lunch: Chicken tikka masala with brown rice and vegetables
                Dinner: Lentil soup with whole-wheat bread
            Day 2
                
                Breakfast: Yogurt with fruit and granola
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables
            Day 3
                
                Breakfast: Eggs with whole-wheat toast and avocado
                Lunch: Salad with grilled chicken or tofu
                Dinner: Stir-fry with vegetables and brown rice
            Day 4
                
                Breakfast: Smoothie made with yogurt, fruit, and spinach
                Lunch: Chickpea curry with rice
                Dinner: Tofu stir-fry with vegetables
            Day 5
                
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
                
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables

        '''
    },
    'Overweight_Level_I': {
        'continental': '''
            Day 1
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Salad with grilled chicken or tofu
                Dinner: Salmon with roasted vegetables
            Day 2
                
                Breakfast: Yogurt with fruit and granola
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Lentil soup with whole-wheat bread
            Day 3
                
                Breakfast: Eggs with whole-wheat toast and avocado
                Lunch: Salad with grilled chicken or tofu
                Dinner: Stir-fry with vegetables and brown rice
            Day 4
                
                Breakfast: Smoothie made with yogurt, fruit, and spinach
                Lunch: Chickpea curry with rice
                Dinner: Tofu stir-fry with vegetables
            Day 5
                
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
                
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables

        ''',
        'Indian': '''
            Day 1

                Breakfast: Oatmeal with berries and nuts
                Lunch: Chicken tikka masala with brown rice and vegetables
                Dinner: Lentil soup with whole-wheat bread
            Day 2
                
                Breakfast: Yogurt with fruit and granola
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables
            Day 3
                
                Breakfast: Eggs with whole-wheat toast and avocado
                Lunch: Salad with grilled chicken or tofu
                Dinner: Stir-fry with vegetables and brown rice
            Day 4
                
                Breakfast: Smoothie made with yogurt, fruit, and spinach
                Lunch: Chickpea curry with rice
                Dinner: Tofu stir-fry with vegetables
            Day 5
                
                Breakfast: Whole-wheat pancakes with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Chicken or tofu kebabs with vegetables
            Day 6
                
                Breakfast: Whole-wheat waffles with fruit
                Lunch: Salad with grilled chicken or tofu
                Dinner: Lentil soup with whole-wheat bread
            Day 7
                
                Breakfast: Oatmeal with berries and nuts
                Lunch: Veggie burger on a whole-wheat bun with salad
                Dinner: Salmon with roasted vegetables
            '''
    }
}

if weight_classification in diet_plan:
    if cuisine in diet_plan[weight_classification]:
        st.write(diet_plan[weight_classification][cuisine])
    else:
        st.write("No diet plan available for the selected cuisine.")
else:
    st.write("Invalid weight classification.")

st.header("Exercise Plan:")

exercise_plan = {
'Insufficient weight':'''
1)Strength training: squats, deadlifts, bench presses, shoulder presses, rows, and pull-ups. Start with lighter weights and gradually increase the resistance as you progress.
2.Resistance training: Bicep curls, tricep extensions, lateral raises, and calf raises are some examples. Aim for higher reps (8-12) and moderate weight to promote muscle hypertrophy.
3.Progressive overload: Gradually increase the intensity, weight, or resistance of your exercises over time to continue challenging your muscles and promoting growth. 
4.Caloric surplus: To gain weight, ensure you consume more calories than you burn. Focus on consuming nutrient-dense foods that are high in protein, carbohydrates, and healthy fats. 
5.Rest and recovery: Allow your body enough time to rest and recover between exercise sessions. This is when your muscles repair and grow stronger. Aim for at least 1-2 rest days per week, and prioritize quality sleep to support the recovery process.
6.Consistency: Consistency is key when it comes to gaining weight and building muscle. Stick to a regular exercise routine and ensure you're consistently consuming a calorie surplus.''',

'Normal_Weight' : '''
1. Cardiovascular exercises: Brisk walking, jogging, running, swimming, cycling, dancing, and playing sports like tennis or basketball. Aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity aerobic activity per week.
2. Strength training:Weightlifting, resistance band exercises, bodyweight exercises like push-ups, squats, lunges, and planks, or using weight machines at the gym. Aim to do strength training exercises at least two days a week.
3. Flexibility and stretching exercises: Static stretching or dynamic stretching, or participate in activities like yoga or Pilates that emphasize flexibility and body awareness.
4. High-intensity interval training (HIIT): HIIT involves alternating between short bursts of intense exercise and periods of active recovery.It include exercises such as jumping jacks, burpees, mountain climbers, or high-knee running.
5. Mind-body exercises: Practices like yoga, tai chi, or meditation can improve flexibility, balance, strength, and mental clarity.
6. Active hobbies and recreational activities: Hiking, dancing, swimming, cycling, gardening, or playing a sport. These activities not only provide exercise but also offer enjoyment and help you maintain an active lifestyle.''',

'Overweight_Level_I':'''
1. Low-impact cardiovascular exercises:Exercises  include brisk walking, swimming, cycling, using an elliptical machine, or water aerobics. Aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity aerobic activity per week.
2. Strength training:  Squats, lunges, push-ups, chest presses, rows, and shoulder presses.
3. Interval training: Interval training involves alternating between high-intensity bursts of exercise and periods of active recovery.  For example, alternate between jogging and walking, or cycling at high intensity for short bursts followed by lower intensity pedaling.
4. Walking or hiking: Aim for brisk walking to increase the intensity and calorie burn. Additionally, hiking on trails or nature paths can provide an enjoyable way to get active and explore the outdoors.
5. Group fitness classes: Joining group fitness classes can provide motivation, support, and a fun exercise environment. Options may include modified aerobics classes, water-based exercises, dance fitness, or strength training classes.
6. Flexibility exercises:  Incorporate gentle stretching exercises into your routine, such as yoga or Pilates, to improve flexibility and promote relaxation.''' , 

'Overweight_Level_II':'''
1. Low-impact cardiovascular exercises: Walking, swimming, water aerobics, cycling, using an elliptical machine, or rowing are good options. Aim for at least 150 minutes of moderate-intensity aerobic activity per week.
2. Strength training: Begin with bodyweight exercises and gradually progress to using resistance bands, dumbbells, or weight machines. Focus on exercises that target major muscle groups such as squats, lunges, push-ups, chest presses, rows, and shoulder presses.
3. Interval training: Interval training involves alternating between high-intensity bursts of exercise and periods of active recovery. Examples include alternating between jogging and walking, or cycling at high intensity for short bursts followed by lower intensity pedaling.
4. Circuit training: Circuit training combines strength training and aerobic exercises in a structured format.  Design a circuit that includes exercises such as squats, lunges, push-ups, planks, step-ups, and jumping jacks.
5. Group fitness classes: Joining group fitness classes can provide motivation, support, and a fun exercise environment. Options may include modified aerobics classes, water-based exercises, dance fitness, or strength training classes.
6. Flexibility exercises: Incorporate gentle stretching exercises into your routine, such as yoga or Pilates, to improve flexibility and promote relaxation.''' , 

"Obesity_Type_I":'''
1. Aerobic exercises: Low-impact options such as walking, swimming, water aerobics, cycling, or using an elliptical machine can be gentle on the joints. Aim for at least 150 minutes of moderate-intensity aerobic activity per week.
2. Strength training:  Begin with bodyweight exercises and gradually progress to using resistance bands, dumbbells, or weight machines. Focus on exercises that target major muscle groups such as squats, lunges, push-ups, chest presses, rows, and shoulder presses.
3. Interval training: Interval training involves alternating between high-intensity bursts of exercise and periods of active recovery. For example, you can alternate between jogging and walking, or cycling at high intensity for short bursts followed by lower intensity pedaling.
4. Group fitness classes: Joining group fitness classes can provide motivation, support, and a fun exercise environment. Options may include Zumba, dance fitness, low-impact aerobics, or modified strength training classes.
5. Flexibility exercises: Incorporate gentle stretching exercises into your routine, such as yoga or Pilates, to improve flexibility and promote relaxation.''' , 

"Obesity_Type_II":'''
1. Low-impact aerobic exercises: Walking, swimming, water aerobics, cycling, using an elliptical machine, or rowing are good options. Aim for at least 150 minutes of moderate-intensity aerobic activity per week.
2. Strength training: Begin with bodyweight exercises and gradually progress to using resistance bands, dumbbells, or weight machines. Focus on exercises that target major muscle groups such as squats, lunges, push-ups, chest presses, rows, and shoulder presses.
3. Interval training: Interval training involves alternating between high-intensity bursts of exercise and periods of active recovery.Examples include alternating between jogging and walking, or cycling at high intensity for short bursts followed by lower intensity pedaling.
4. Circuit training: Circuit training combines strength training and aerobic exercises in a structured format. Design a circuit that includes exercises such as squats, lunges, push-ups, planks, step-ups, and jumping jacks.
5. Functional training:Examples include bodyweight squats, standing lunges, step-ups, stair climbing, carrying groceries, or using resistance bands for resistance-based movements.
6. Group fitness classes: Joining group fitness classes can provide motivation, support, and a fun exercise environment. Options may include Zumba, dance fitness, low-impact aerobics, or modified strength training classes.''' , 

"Obesity_Type_III":'''
1. Low-impact aerobic exercises: These can include walking, stationary cycling, using an elliptical machine, or water aerobics. Start with shorter durations and gradually increase as your fitness level improves.
2. Strength training: Begin with bodyweight exercises and gradually progress to using resistance bands, dumbbells, or weight machines. Focus on exercises that target major muscle groups such as squats, lunges, push-ups, chest presses, and rows.
3. Chair exercises: For individuals with limited mobility or difficulty with weight-bearing exercises, chair exercises can be beneficial. Examples include seated leg extensions, seated marches, seated upper body exercises with resistance bands, or seated abdominal crunches.
4. Flexibility and stretching exercises: Flexibility exercises help improve joint mobility, reduce muscle stiffness. Incorporate gentle stretching exercises, such as yoga or tai chi, to improve flexibility and promote relaxation.
5. Mind-body exercises: Mind-body exercises, such as yoga or Pilates, can help improve strength, flexibility, and body awareness. These exercises focus on controlled movements, breathing techniques, and mindfulness, which can be beneficial for individuals with obesity Type 3.''' , 

}

if weight_classification in exercise_plan:
        st.write(exercise_plan[weight_classification])
else:
    st.write("Invalid weight classification.")