metrics_age_types=["в годах","в месецах"]
gender_types=["Самец", "Самка"]
rep_status_types=["Не беременная", "Беременная", "Период лактации"]
berem_time_types=["первые 4 недедели беременности","последние 5 недель беременности"]
lact_time_types=["1 неделя","2 неделя","3 неделя","4 неделя"]
age_category_types=["Щенки","Взрослые","Пожилые"]
size_types=["Мелкие",  "Средние",  "Крупные", "Очень крупные"]
activity_level_cat_1 = ["Пассивный (гуляеет на поводке менее 1ч/день)", "Средний1 (1-3ч/день, низкая активность)",
                          "Средний2 (1-3ч/день, высокая активность)", "Активный (3-6ч/день, рабочие собаки, например, овчарки)",
                          "Высокая активность в экстремальных условиях (гонки на собачьих упряжках со скоростью 168 км/день в условиях сильного холода)",
                          "Взрослые, склонные к ожирению"]
activity_level_cat_2 = ["Пассивный", "Средний", "Активный"]


other_nutrients=["Зола","Клетчатка","Холестерин, mg","Марганец, mg","Селен, mkg","Сахар общее","Тиамин, mg"]
major_minerals=["Major Minerals.Calcium, mg","Major Minerals.Copper, mg","Major Minerals.Iron, mg","Major Minerals.Magnesium, mg","Major Minerals.Phosphorus, mg","Major Minerals.Potassium, mg",
                "Major Minerals.Sodium, mg","Major Minerals.Zinc, mg"]
vitamins=["Vitamin A - IU, mkg","Vitamin A - RAE, mkg","Vitamin B12, mkg","Vitamin B6, mkg","Vitamin C, mkg","Vitamin E, mkg","Vitamin K, mkg"]
 
def kcal_calculate(reproductive_status, berem_time, num_pup, L_time, age_type, weight, expected, activity_level, user_breed, age):
    formula=""
    page=""
    if L_time==lact_time_types[0]:
      L=0.75
    elif L_time==lact_time_types[1]:
      L=0.95
    elif L_time==lact_time_types[2]:
      L=1.1
    else :
      L=1.2
    
    if reproductive_status==rep_status_types[1]:
      if berem_time==berem_time_types[0]:
        kcal=132*(weight**0.75)
        formula= r"kcal = 132 \cdot вес^{0.75}  \\  \text{(первые 4 недели беременности)}"
        page = "56"
        
      else:
        kcal=132*(weight**0.75) + (26*weight)
        formula= r"kcal = 132 \cdot вес^{0.75} + 26 \cdot вес  \\  \text{(последние 5 недель беременности)}"
        page = "56"
  
    elif reproductive_status==rep_status_types[2]:
       if num_pup<5:
         kcal=145*(weight**0.75) + 24*num_pup*weight*L
         formula = fr"kcal = 145 \cdot вес^{{0.75}} + 24 \cdot n \cdot вес \cdot L  \\  \text{{n - количество щенков}}  \\  \text{{L = {L} для {L_time}}}"
         page = "56"
         
       else:
         kcal=145*(weight**0.75) + (96+12*num_pup-4)*weight*L
         formula = fr"kcal = 145 \cdot вес^{{0.75}}  + (96 + 12 \cdot n - 4) \cdot вес \cdot L    \\  \text{{n - количество щенков}}  \\  \text{{L = {L} для {L_time}}}"       
         page = "56"
         
    else:
      if age_type==age_category_types[0]:
          if age<8:
            kcal=25 * weight 
            formula= r"kcal = 25 \cdot вес"
            page = "56"
            
          elif age>=8 and age <12:
            kcal=(254.1-135*(weight/expected) )*(weight**0.75)
            formula=fr"kcal = \left(254.1 - 135 \cdot \frac{{вес}}{{w}}\right) \cdot вес^{{0.75}}  \\  w = {round(expected,2)}  \text{{кг ;  предположительный вес для породы {user_breed}}}"
            page = "56"

        
          else :
            kcal=130*(weight**0.75)
            formula= r"kcal = 130 \cdot вес^{0.75}"
            page = "54"


      
      elif age_type==age_category_types[2]:
          if activity_level==activity_level_cat_2[0]:
              kcal=80*(weight**0.75)
              formula= r"kcal = 80  \cdot вес^{0.75}"
              page = "54"
        
          elif activity_level==activity_level_cat_2[1]:
              kcal=95*(weight**0.75)
              formula= r"kcal = 95  \cdot вес^{0.75}"
              page = "54"    
            
          else:
             kcal=110*(weight**0.75)
             formula= r"kcal = 110  \cdot вес^{0.75}"
             page = "54"
            
      else:   
            if activity_level==activity_level_cat_1[0]:
              kcal=95*(weight**0.75)
              formula= r"kcal = 95  \cdot вес^{0.75}"
              page = "55"
        
            elif activity_level==activity_level_cat_1[1]:
              kcal=110*(weight**0.75)
              formula= r"kcal = 110  \cdot вес^{0.75}"
              page = "55"
        
            elif activity_level==activity_level_cat_1[2]:
              kcal=125*(weight**0.75)
              formula= r"kcal = 125  \cdot вес^{0.75}"
              page = "55"
        
            elif activity_level==activity_level_cat_1[3]:
              kcal=160*(weight**0.75)
              formula= r"kcal = 160  \cdot вес^{0.75}"
              page = "55"
              
            elif activity_level==activity_level_cat_1[4]:
              kcal=860*(weight**0.75)
              formula= r"kcal = 860  \cdot вес^{0.75}"
              page = "55"
           
            else:
              kcal=90*(weight**0.75)
              formula= r"kcal = 90  \cdot вес^{0.75}"
              page = "55"
              
    return kcal, formula, page
