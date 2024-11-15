import pandas as pd
import re

# Read the text file with ' ::: ' as the delimiter, no headers initially
df = pd.read_csv(r"Genre Classification Dataset\test_data.txt", sep=" ::: ", engine="python", header=None)

# Set column names
df.columns = ["ID", "Title", "Genre", "Description"]

# Remove the year from the Title using a regular expression
df["Title"] = df["Title"].apply(lambda x: re.sub(r"\s*\(\d{4}\)$", "", x))

# Save to CSV
df.to_csv("train_data.csv", index=False)

print("Conversion complete. Data saved to train_data.csv")



df1 = pd.read_csv(r"Genre Classification Dataset\test_data.txt", sep=" ::: ", engine="python", header=None)

df1.columns = ["ID", "Title", "Description"]

df1["Title"] = df1["Title"].apply(lambda x: re.sub(r"\s*\(\d{4}\)$", "", x))


df1.to_csv("test_data.csv", index=False)

print("Conversion complete. Data saved to test_data.csv")

