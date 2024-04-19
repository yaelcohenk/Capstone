def category(df):
    a = 0
    
    if((df['ADI']<=1.34) & (df['CV2']<=0.49)):
        a='Smooth'
    if((df['ADI']>=1.34) & (df['CV2']>=0.49)):  
        a='Lumpy'
    if((df['ADI']<1.34) & (df['CV2']>0.49)):
        a='Erratic'
    if((df['ADI']>1.34) & (df['CV2']<0.49)):
        a='Intermittent'
    return a