# import pandas as pd
# import matplotlib.pyplot as plt

# # Load dataset
# file_path = "C:/Users/MSI/Downloads/all_responses_clean-all_responses_clean (1).csv"  # update path if needed
# df = pd.read_csv(file_path)

# # --- Sentiment Distribution ---
# sentiment_distribution = df['Sentiment'].value_counts()

# plt.figure(figsize=(6,4))
# sentiment_distribution.plot(kind="bar", color=["green","red","gray"])
# plt.title("Sentiment Distribution")
# plt.xlabel("Sentiment")
# plt.ylabel("Count")
# plt.xticks(rotation=0)
# plt.show()

# # --- Gender Distribution ---
# gender_distribution = df['Gender'].value_counts()

# plt.figure(figsize=(5,5))
# plt.pie(gender_distribution, labels=gender_distribution.index, autopct="%1.1f%%", startangle=90)
# plt.title("Gender Distribution")
# plt.show()

# # --- Empathy Score Distribution ---
# plt.figure(figsize=(6,4))
# plt.hist(df["Empathy"].dropna(), bins=5, edgecolor="black")
# plt.title("Empathy Score Distribution")
# plt.xlabel("Empathy Score")
# plt.ylabel("Frequency")
# plt.show()

# # --- Advice Quality Distribution ---
# plt.figure(figsize=(6,4))
# plt.hist(df["AdviceQuality"].dropna(), bins=5, edgecolor="black", color="orange")
# plt.title("Advice Quality Score Distribution")
# plt.xlabel("Advice Quality Score")
# plt.ylabel("Frequency")
# plt.show()


# # --- GenderedLang ---
# # plt.figure(figsize=(6,4))
# # plt.hist(df["GenderedLang"].dropna(), bins=5, edgecolor="black", color="purple")
# # plt.title("Gendered Language Score Distribution")
# # plt.xlabel("Gendered Language Score")
# # plt.ylabel("Frequency")
# # plt.show()


# gender_Lang_distribution = df['GenderedLang'].value_counts()

# plt.figure(figsize=(5,5))
# plt.pie(gender_Lang_distribution, labels=gender_Lang_distribution.index, autopct="%1.1f%%", startangle=90)
# plt.title("Gendered Language Distribution")
# plt.show()




# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load dataset
# file_path = "C:/Users/MSI/Downloads/all_responses_clean-all_responses_clean (1).csv"  # update path if needed
# df = pd.read_csv(file_path)

# # Set modern style
# sns.set_theme(style="whitegrid")

# # ================= Sentiment Distribution =================
# sentiment_distribution = df['Sentiment'].value_counts(normalize=False)
# sentiment_percent = df['Sentiment'].value_counts(normalize=True) * 100

# plt.figure(figsize=(7,5))
# ax = sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values, palette="Set2")

# # Add counts + percentages on bars
# for i, (count, pct) in enumerate(zip(sentiment_distribution.values, sentiment_percent.values)):
#     ax.text(i, count + 1, f"{count} ({pct:.1f}%)", ha="center", fontsize=11, fontweight="bold")

# plt.title("Sentiment Distribution", fontsize=14, fontweight="bold")
# plt.xlabel("Sentiment")
# plt.ylabel("Count")
# plt.show()

# # ================= Gender Distribution =================
# gender_distribution = df['Gender'].value_counts()
# gender_percent = df['Gender'].value_counts(normalize=True) * 100

# plt.figure(figsize=(6,6))
# colors = sns.color_palette("pastel")
# plt.pie(gender_distribution, labels=[f"{g} ({p:.1f}%)" for g,p in zip(gender_distribution.index, gender_percent)],
#         autopct="", startangle=90, colors=colors, wedgeprops={'edgecolor':'black'})
# plt.title("Gender Distribution", fontsize=14, fontweight="bold")
# plt.show()

# # ================= Empathy Score Distribution =================
# plt.figure(figsize=(7,5))
# sns.histplot(df["Empathy"].dropna(), bins=5, kde=True, color="skyblue", edgecolor="black")
# plt.title("Empathy Score Distribution", fontsize=14, fontweight="bold")
# plt.xlabel("Empathy Score")
# plt.ylabel("Frequency")
# plt.show()

# # ================= Advice Quality Distribution =================
# plt.figure(figsize=(7,5))
# sns.histplot(df["AdviceQuality"].dropna(), bins=5, kde=True, color="orange", edgecolor="black")
# plt.title("Advice Quality Score Distribution", fontsize=14, fontweight="bold")
# plt.xlabel("Advice Quality Score")
# plt.ylabel("Frequency")
# plt.show()


from typing import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "scored_responses.csv"  # update path if needed
df = pd.read_csv(file_path)

sns.set_theme(style="whitegrid")

# ================= Sentiment by Gender =================
plt.figure(figsize=(7,5))
ax = sns.countplot(data=df, x="sentiment_score", hue="Gender", palette="Set2")

# Add labels
for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(count, (p.get_x() + p.get_width() / 2., count),
                ha='center', va='center', fontsize=10, fontweight="bold", xytext=(0,5), textcoords='offset points')

plt.title("Sentiment Distribution by Gender", fontsize=14, fontweight="bold")
plt.xlabel("sentiment_score")
plt.ylabel("Count")
plt.legend(title="Gender")
plt.show()

# ================= Empathy by Gender =================
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="Gender", y="empathy_score", palette="pastel")
sns.stripplot(data=df, x="Gender", y="empathy_score", color="black", alpha=0.5)
plt.title("Empathy Score by Gender", fontsize=14, fontweight="bold")
plt.show()

# ================= Advice Quality by Gender =================
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="Gender", y="advice_quality_score", palette="Set3")
sns.stripplot(data=df, x="Gender", y="advice_quality_score", color="black", alpha=0.5)
plt.title("Advice Quality by Gender", fontsize=14, fontweight="bold")
plt.show()

# ================= Gender Counts =================
plt.figure(figsize=(6,6))
gender_counts = df['Gender'].value_counts()
gender_percent = df['Gender'].value_counts(normalize=True) * 100

colors = sns.color_palette("pastel")
plt.pie(gender_counts, labels=[f"{g} ({p:.1f}%)" for g,p in zip(gender_counts.index, gender_percent)],
        autopct="", startangle=90, colors=colors, wedgeprops={'edgecolor':'black'})
plt.title("Gender Distribution", fontsize=14, fontweight="bold")
plt.show()


# ================= Word Count by Gender =================
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="Gender", y="word_count", palette="Set2")
sns.stripplot(data=df, x="Gender", y="word_count", color="black", alpha=0.5)
plt.title("Word Count Distribution by Gender", fontsize=14, fontweight="bold")
plt.xlabel("Gender")
plt.ylabel("Word Count")
plt.show()

# ================= Biased Terms Frequency by Gender =================
# Split biased terms into individual words
biased_terms_by_gender = {}
for gender in df['Gender'].unique():
    terms = df[df['Gender'] == gender]['gendered_terms'].dropna().str.cat(sep=",").split(",")
    terms = [t.strip().lower() for t in terms if t.strip()]
    biased_terms_by_gender[gender] = Counter(terms)

# Prepare top 10 terms for each gender
plot_data = []
for gender, counter in biased_terms_by_gender.items():
    for term, count in counter.most_common(10):
        plot_data.append({"Gender": gender, "gendered_terms": term, "Count": count})

plot_df = pd.DataFrame(plot_data)

plt.figure(figsize=(12,6))
ax = sns.barplot(x="gendered_terms", y="Count", hue="Gender", data=plot_df, palette="Set1")

# Add labels on bars
for p in ax.patches:
    height = int(p.get_height())
    if height > 0:
        ax.annotate(height,
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', fontsize=10, fontweight="bold",
                    xytext=(0,5), textcoords='offset points')

plt.title("Top 10 Biased Terms Frequency by Gender", fontsize=14, fontweight="bold")
plt.xlabel("gendered_terms")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.legend(title="Gender")
plt.show()
