"""
Author: Garrett Powell & Gale Proulx
Class: Advanced Data Analytics
Assignment: Final Project
Date: 25 February 2020

Certification of Authenticity:

I certify that this is entirely my own work, except where I have given fully-documented references to the work of
others. I understand the definition and consequences of plagiarism and acknowledge that the assessor of this assignment
may, for the purpose of assessing this assignment, reproduce this assignment and provide a copy to another member of
academic staff; and/or-Communicate a copy of this assignment to a plagiarism checking service (which may then retain a
copy of this assignment on its database for the purpose of future plagiarism checking).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_questions(data: pd.DataFrame):
    return data.drop_duplicates(["question_code"])[["question_code", "question_label"]]


def convert_dtype(data: pd.DataFrame) -> pd.DataFrame:
    copy = data.copy()
    copy["percentage"] = pd.to_numeric(copy["percentage"], errors="coerce")
    return copy


def main():
    daily_life = convert_dtype(pd.read_csv("LGBT_Survey_DailyLife.csv"))
    discrimination = convert_dtype(pd.read_csv("LGBT_Survey_Discrimination.csv"))
    rights = convert_dtype(pd.read_csv("LGBT_Survey_RightsAwareness.csv"))
    trans = convert_dtype(pd.read_csv("LGBT_Survey_TransgenderSpecificQuestions.csv"))
    violence = convert_dtype(pd.read_csv("LGBT_Survey_ViolenceAndHarassment.csv"))

    questions = get_questions(daily_life)

    # What groups do people tend to be out to the most?
    groups = daily_life[daily_life["question_code"].str.startswith("g3")]
    group_averages = groups.groupby(["question_code", "answer"])["percentage"].mean().reset_index()
    group_data = pd.DataFrame({
        "Group": group_averages["question_code"].map({
            "g3_a": "Family",
            "g3_b": "Friends",
            "g3_c": "Neighbors",
            "g3_d": "Colleagues/Classmates",
            "g3_e": "Superior/Boss",
            "g3_f": "Customers/Clients",
            "g3_g": "Medical Staff"
        }),
        "Answer": pd.Categorical(
            group_averages["answer"],
            ["None", "A few", "Most", "All"]
        ),
        "Percentage": group_averages["percentage"]
    })
    sns.barplot(data=group_data, x="Group", y="Percentage", hue="Answer")
    plt.title("Groups That LGBT People Are Out To")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/Groups That LGBT People Are Out To.png", dpi=300)
    plt.clf()

    # Which groups are the most discriminated against?
    groups = daily_life[daily_life["question_code"].str.startswith("c1a")]
    group_averages = groups.groupby(["question_code", "answer"])["percentage"].mean().reset_index()
    group_data = pd.DataFrame({
        "Identity": group_averages["question_code"].map({
            "c1a_a": "Lesbian",
            "c1a_b": "Gay",
            "c1a_c": "Bisexual",
            "c1a_d": "Transgender",
        }),
        "Answer": pd.Categorical(
            group_averages["answer"],
            ["Don`t know", "Very rare", "Fairly rare", "Fairly widespread", "Very widespread"]
        ),
        "Percentage": group_averages["percentage"]
    })
    sns.barplot(data=group_data, x="Identity", y="Percentage", hue="Answer")
    plt.title("How Widespread LGBT People Feel Discrimination Is")
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/How Widespread LGBT People Feel Discrimination Is.png", dpi=300)
    plt.clf()

    # What places do LGBT people avoid for fear of violence?
    groups = violence[violence["question_code"] == "e3"]
    group_averages = groups.groupby("answer")["percentage"].mean().reset_index()
    group_averages = group_averages.sort_values("percentage", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=group_averages, x="percentage", y="answer", orient="h")
    plt.title("Places LGBT People Avoid For Fear of Violence")
    plt.xlabel("Percentage")
    plt.ylabel("Location")
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/Places LGBT People Avoid For Fear of Violence.png", dpi=300)
    plt.clf()

    # Why don't LGBT people report violence?
    groups = violence[violence["question_code"] == "fa1_12"]
    group_averages = groups.groupby("answer")["percentage"].mean().reset_index()
    group_averages = group_averages.sort_values("percentage", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=group_averages, x="percentage", y="answer", orient="h")
    plt.title("Why LGBT People Don't Report Acts of Violence")
    plt.xlabel("Percentage")
    plt.ylabel("Reason")
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/Why LGBT People Don't Report Acts of Violence.png", dpi=300)
    plt.clf()

    # Why don't transgender people seek help?
    groups = trans[trans["question_code"] == "tr2"]
    group_averages = groups.groupby("answer")["percentage"].mean().reset_index()
    group_averages = group_averages.sort_values("percentage", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=group_averages, x="percentage", y="answer", orient="h")
    plt.title("Why Transgender People Don't Seek Help")
    plt.xlabel("Percentage")
    plt.ylabel("Reason")
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/Why Transgender People Don't Seek Help.png", dpi=300)
    plt.clf()

    # What Would Make Living as a Transgender Person Easier?
    groups = trans[trans["question_code"].str.startswith("tr6")]
    group_averages = groups.groupby(["question_code", "answer"])["percentage"].mean().reset_index()
    group_data = pd.DataFrame({
        "Measure": group_averages["question_code"].map({
            "tr6_a": "More options for medical treatment",
            "tr6_b": "Easier legal gender change",
            "tr6_c": "Workplace anti-discrimination policies",
            "tr6_d": "Measures implemented at school",
            "tr6_e": "Support from public figures",
            "tr6_f": "Support from national authorities",
            "tr6_g": "Training of public servants",
            "tr6_h": "Acceptance from religious leaders"
        }),
        "Answer": pd.Categorical(
            group_averages["answer"],
            ["Don`t know", "Strongly disagree", "Disagree", "Current situation is fine", "Agree", "Strongly agree"]
        ),
        "Percentage": group_averages["percentage"]
    })
    plt.figure(figsize=(10, 10))
    sns.barplot(data=group_data, x="Percentage", y="Measure", hue="Answer", orient="h")
    plt.title("Measures That Would Make Living as a Transgender Person Easier")
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/Measures That Would Make Living as a Transgender Person Easier.png", dpi=300)
    plt.clf()

    # How satisfied are LGBT people with their lives across the EU?
    groups = daily_life[daily_life["question_code"] == "g5"]
    group_averages = groups.groupby(["CountryCode", "answer"])["percentage"].mean().reset_index()
    group_averages["satisfaction"] = group_averages["answer"].astype("float64") * (group_averages["percentage"] / 100)
    country_data = group_averages.groupby("CountryCode")["satisfaction"].sum()
    country_data = country_data.sort_values(ascending=False)
    sns.barplot(x=country_data, y=country_data.index, orient="h")
    plt.title("Satisfaction of LGBT People By Country")
    plt.xlabel("Satisfaction (1-10)")
    plt.ylabel("Country")
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/Satisfaction of LGBT People By Country.png", dpi=300)
    plt.clf()

    return


if __name__ == "__main__":
    main()
