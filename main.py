import data_cleaning
import training

df = data_cleaning.clean_data()
training.train(df)
