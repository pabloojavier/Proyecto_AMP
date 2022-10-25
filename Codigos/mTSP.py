import pandas as pd

def read_data(ins):
    if ins in [51,100,150]:
        #coords = pd.read_excel("Instancias/51 - 100 - 150 city prolems.xls","100Coords",header=0,index_col=0)
        return pd.read_excel("Instancias/51 - 100 - 150 city prolems.xls",f"{ins}Dist",header=None)
    else:
        return pd.read_csv(f"Instancias/{ins}_dist.csv",header=None)
    
instancias = ["sp11","uk12",51,100,"sgb128",150]


for i in instancias:
    print(read_data(i))
