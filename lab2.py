import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
file_path = r"C:\Users\fayad\Downloads\Automobile - Automobile.csv"
df = pd.read_csv(file_path)
if 'horsepower' in df.columns:
    df.drop(columns=['horsepower'], inplace=True)
    
imputer = SimpleImputer(strategy='median')
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

df_minmax.to_csv(r"C:\Users\fayad\Downloads\Automobile_minmax_scaled_0141.csv", index=False)
df_standard.to_csv(r"C:\Users\fayad\Downloads\Automobile_standard_scaled_0141.csv", index=False)
print("Data preprocessing completed! Files saved in Downloads folder.")
