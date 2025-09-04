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



import streamlit as st
import matplotlib.pyplot as plt

def bar_print(total_norm,current_value,name_ing,mg):
                                        maxi_dat = total_norm if total_norm>current_value else current_value
                                        norma = 100 if maxi_dat== total_norm else (total_norm/current_value)*100
                                        curr =  100 if maxi_dat== current_value else (current_value/total_norm)*100
  
                                        maxi_lin = 100*1.2
                                        diff = current_value - total_norm
                                        fig, ax = plt.subplots(figsize=(5, 1))
                                        ax.axis('off')
                                        # Добавляем запас 20% справа и фиксируем начало оси X
                                        ax.set_xlim(-50, maxi_lin+8)
                                        ax.set_ylim(-0.5, 0.5)
                                        ax.plot([0, maxi_lin], [0, 0], color='#e0e0e0', linewidth=20, solid_capstyle='round', alpha=0.8)
                                        fixed_space = -10 
                                        ax.text(fixed_space, 0, name_ing, ha='right', va='center', fontsize=13, fontweight='bold')
                                        if current_value < total_norm:
                                            ax.plot([0, norma], [0, 0], color='green', linewidth=20, solid_capstyle='round')
                                            ax.plot([0, curr], [0, 0], color='purple', linewidth=20, solid_capstyle='round')
                                        else:
                                            ax.plot([0, curr], [0, 0], color='red', linewidth=20, solid_capstyle='round')
                                            ax.plot([0, norma], [0, 0], color='green', linewidth=20, solid_capstyle='round')
                                        ax.text(maxi_lin+10, 0,
                                                f"{'Дефицит' if diff < 0 else 'Избыток'}: {round(abs(diff),1)} {mg}",
                                                ha='left', va='center', fontsize=13, color='black')
                                        ax.text(curr, 0.2, f"Текущее\n{current_value}", color='purple', ha='center', va='bottom', fontsize=9)
                                        ax.text(norma, -0.2,  f"Норма\n{total_norm}", color='green', ha='center', va='top', fontsize=9)
                                        return fig

def size_category(w):
    if w <= 10:
        return size_types[0]
    elif w <= 25:
        return size_types[1]
    elif w <= 40:
        return size_types[2]
    else:
        return size_types[3]

def age_type_category(size_categ, age ,age_metric):
        if age_metric==metrics_age_types[0]:
            age=age*12
            
        if size_categ==size_types[0]:
          if age>=1*12 and age<=8*12:    
             return age_category_types[1]
          elif age<1*12:    
             return age_category_types[0]
          elif age>8*12:  
             return age_category_types[2]
       
        elif size_categ==size_types[2]:
          if age>=15 and age<=7*12  :   
              return age_category_types[1]
          elif age<15:     
             return age_category_types[0]
          elif age>7*12:    
             return age_category_types[2]
              
        elif size_categ==size_types[3]:
          if age<=6*12 and age>=24:    
              return age_category_types[1]
          elif age<24:    
              return age_category_types[0]
          elif age>6*12:   
              return age_category_types[2]
              
        else:  
          if age<=7*12:
                return age_category_types[1]
          elif age<12:     
             return age_category_types[0]
          elif age>7*12:    
            return age_category_types[2]
            
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
