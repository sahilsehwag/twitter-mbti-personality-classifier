# Introduction
This is a *machine learning* project. This trained machine learning classifier can predict a person's **MBTI**(Myers–Briggs Type Indicator) 
personality type using an individual's *social media posts* like **twitter posts**. 

### About MBTI
The Myers Briggs Type Indicator (or MBTI for short) is a personality type system that divides everyone into 16 distinct personality types across 4 axis:
* **Introversion** (I)/**Extroversion** (I)
* **Intuition** (N)/**Sensing** (S)
* **Thinking** (T)/**Feeling** (F)
* **Judging** (J)/**Percieving** (P)
<br>
You can read more about the MBTI test [here](https://en.wikipedia.org/wiki/Myers%E2%80%93Briggs_Type_Indicator).

### Dataset
The [dataset](./personality-test.csv) on which this classifier is trained, contains around 50 posts per user about 8000 users with their *MBTI* 
personality type known. Dataset is provided in the repo itself [personality-test.csv](./personality-test.csv).
<br>
Few features of dataset are:
* Posts of more than 8000 users.
* Last 50 posts per user, each entry is separated by '|||'.

# Usage
