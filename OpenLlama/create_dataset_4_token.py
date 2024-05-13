import json
import random
import numpy as np
all_text=[]
with open('data/MIdataset/single.json','r',encoding='utf-8')as f:
    all_data=json.load(f)
    num=0
    text_sample=''
    for i_1 in range(15):
        for i_2 in range(15):  
            if i_2!=i_1:
                  for i_3   in range(15):  
                          if i_3!=i_1 and i_3 !=i_2:
                                  for i_4   in range(15):  
                                        if i_4!=i_1 and i_4 !=i_2 and i_4 !=i_3:
                                            for i_5 in range(15):
                                                if i_5!=i_1 and i_5 !=i_2 and i_5 !=i_3 and i_5 !=i_4:
                                                    index=np.arange(16)
                                                    
                                                    np.random.shuffle(index)
                                                    text_sample=text_sample+all_data[index[0]]['name']+', '+all_data[index[1]]['name']+', '+all_data[index[2]]['name']+', '+all_data[index[3]]['name']+', '
                                                    all_text.append(text_sample)
                                                    num=num+1
                                                    text_sample=''
    print(num)
with open('data/MIdataset/all_country_8_token.json','w',encoding='utf-8')as f:
        json.dump(all_text,f,ensure_ascii=False,sort_keys=True)      
        
                           