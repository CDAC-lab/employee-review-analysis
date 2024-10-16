# -*- coding: utf-8 -*-
"""Employee Review Analysis - Discourse Analysis

***Install and import packages***
"""

!pip install --upgrade openai pandas

"""***Identity - Sentiment***"""

import pandas as pd
from openai import OpenAI
import os
import time

# Initialize OpenAI client with your API key
client = OpenAI(
    api_key='.......',
)

def analyse_user_prompt(text):
    time.sleep(1)
    system_msg = """You are an expert in discourse analysis and psychology.


    Each of reviews have been posted by current and former employees. Based on the analysis, identify whether the employee is with positive, negative or neutral sentiment?
    Mention only the relevant category as the answer

    Example of an analysis for your reference.
    Review -{I had a terrible experience when handling customers, my manager hardly trained me on how to do much of anything so when I had an actual customer I would stutter and stagger. She didn't bother to help me when handling customers so when I made a mistake she wrote it down and sent a report that I was disobeying when in reality I was just confused on some small things here and there. I did not enjoy my time working under that employer at the time.}
    Answer - Negative sentiment

    You can extract the answer to the CSV file as one column for ID and another column for the answer
          """

    messages=[
        {
          "role": "system",
          "content": system_msg
        },
        {
          "role": "user",
          "content": text
        }]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content

# Read reviews from CSV file
input_csv_path = 'File_Name.csv'  # Adjust the path to your input CSV file
df_reviews = pd.read_csv(input_csv_path)

# Analyze each review
identities = []
for review in df_reviews['review']:
    result = analyse_user_prompt(review)
    identities.append(result)  # Collect results

# Create a DataFrame for the results
df_results = pd.DataFrame({
    'review': df_reviews['review'],
    'identities': identities
})

# Save results to a new CSV file
output_csv_path = 'identities_results.csv'  # Adjust the path for output CSV file
df_results.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")

"""***Relationship***"""

import pandas as pd
from openai import OpenAI
import os
import time

# Initialize OpenAI client with your API key
client = OpenAI(
    api_key='.......',
)

def analyse_user_prompt(text):
    time.sleep(1)
    system_msg = """You are an expert in discourse analysis and psychology.


    Each of reviews have been posted by current and former employees. Based on the analysis, identify whether what type of relationship employee represents in the review.
    One review may present more than one relationship type.
    1.	Remote Work Relationships
    2.	Peer Relationships
    3.	Toxic Relationships
    4.	Dysfunctional Team Relationships
    5.	Manager and Employee Relationships
    6.	Customer Relationships
    7.	Cross-Departmental Relationships
    8.	Supplier-Vendor Relationships
    9.	Professional and Development Relationships
    10.	Human Resources Relationships
    11.	Formal Relationships
    12.	Reward-Based Relationships
    13.	Project-Based Relationships
    14.	Mentor-Mentee Relationships
    15.	Networking Relationships
    16.	Micromanagement Relationships
    17.	Bullying Relationships
    18.	Recruitment Relationships
    19.	Business Relationships
    20.	Work-Life Balance Relationships
    21.	Discrimination Relationships
    22.	Romantic Relationships


    Mention only the relevant category or categories as the answer

    Example of an analysis for your reference.
    Review -{I had a terrible experience when handling customers, my manager hardly trained me on how to do much of anything so when I had an actual customer I would stutter and stagger. She didn't bother to help me when handling customers so when I made a mistake she wrote it down and sent a report that I was disobeying when in reality I was just confused on some small things here and there. I did not enjoy my time working under that employer at the time.}
    Answer - Manager and Employee Relationships, Customer Relationships

    You can extract the answe to the CSV file as one column for ID and another column for the answer
          """

    messages=[
        {
          "role": "system",
          "content": system_msg
        },
        {
          "role": "user",
          "content": text
        }]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content

# Read reviews from CSV file
input_csv_path = 'File_Name.csv'  # Adjust the path to your input CSV file
df_reviews = pd.read_csv(input_csv_path)

# Analyze each review
relationships = []
for review in df_reviews['review']:
    result = analyse_user_prompt(review)
    relationships.append(result)  # Collect results

# Create a DataFrame for the results
df_results = pd.DataFrame({
    'review': df_reviews['review'],
    'relationships': relationships
})

# Save results to a new CSV file
output_csv_path = 'relationships_results.csv'  # Adjust the path for output CSV file
df_results.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")

"""***Power***"""

import pandas as pd
from openai import OpenAI
import os
import time

# Initialize OpenAI client with your API key
client = OpenAI(
    api_key='.......',
)

def analyse_user_prompt(text):
    time.sleep(1)
    system_msg = """You are an expert in discourse analysis and psychology.


    Each of reviews have been posted by current and former employees. Based on the analysis, identify whether what type of power employee has experienced as mentioned in the review.
    One review may present more than one type of power.
    1.	High coercive power
    2.	Medium coercive power
    3.	Low coercive power
    4.	High reward power
    5.	Medium reward power
    6.	Low reward power
    7.	High legitimate power
    8.	Medium legitimate power
    9.	Low legitimate power
    10.	High referent power
    11.	Medium referent power
    12.	Low referent power
    13.	High Expert power
    14.	Medium Expert power
    15.	Low Expert power


    Example of an analysis for your reference.
    Review -{I had a terrible experience when handling customers, my manager hardly trained me on how to do much of anything so when I had an actual customer I would stutter and stagger. She didn't bother to help me when handling customers so when I made a mistake she wrote it down and sent a report that I was disobeying when in reality I was just confused on some small things here and there. I did not enjoy my time working under that employer at the time.}
    Answer - High coercive power, Low expert power, Low reward power, Low legitimate power

    You can extract the answe to the CSV file as one column for ID and another column for the answer
          """

    messages=[
        {
          "role": "system",
          "content": system_msg
        },
        {
          "role": "user",
          "content": text
        }]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content

# Read reviews from CSV file
input_csv_path = 'File_Name.csv'  # Adjust the path to your input CSV file
df_reviews = pd.read_csv(input_csv_path)

# Analyze each review
power = []
for review in df_reviews['review']:
    result = analyse_user_prompt(review)
    power.append(result)  # Collect results

# Create a DataFrame for the results
df_results = pd.DataFrame({
    'review': df_reviews['review'],
    'power': power
})

# Save results to a new CSV file
output_csv_path = 'power.csv'  # Adjust the path for output CSV file
df_results.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")

"""***Context***"""

import pandas as pd
from openai import OpenAI
import os
import time

# Initialize OpenAI client with your API key
client = OpenAI(
    api_key='.......',
)

def analyse_user_prompt(text):
    time.sleep(1)
    system_msg = """You are an expert in discourse analysis and psychology.


    Each of reviews have been posted by current and former employees. Based on the analysis, identify how the employee has experienced the context of the organisation.
    One review may present more than one type of contexts
    1.	People And Relationships
    2.	Teamwork
    3.	Social Climate
    4.	Work Organisation
    5.	Work Control And Flexibility
    6.	Growth And Rewards
    7.	Purpose
    8.	Technology
    9.	Physical Environment
    10.	Leadership


    Example of an analysis for your reference.
    Review -{I had a terrible experience when handling customers, my manager hardly trained me on how to do much of anything so when I had an actual customer I would stutter and stagger. She didn't bother to help me when handling customers so when I made a mistake she wrote it down and sent a report that I was disobeying when in reality I was just confused on some small things here and there. I did not enjoy my time working under that employer at the time.}
    Answer - people and relationships, teamwork, social climate, work organization, work control and flexibility, growth and rewards, purpose

    You can extract the answer to the CSV file as one column for ID and another column for the answer
          """

    messages=[
        {
          "role": "system",
          "content": system_msg
        },
        {
          "role": "user",
          "content": text
        }]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content

# Read reviews from CSV file
input_csv_path = 'File_Name.csv'  # Adjust the path to your input CSV file
df_reviews = pd.read_csv(input_csv_path)

# Analyze each review
context = []
for review in df_reviews['review']:
    result = analyse_user_prompt(review)
    context.append(result)  # Collect results

# Create a DataFrame for the results
df_results = pd.DataFrame({
    'review': df_reviews['review'],
    ' context':  context
})

# Save results to a new CSV file
output_csv_path = ' context.csv'  # Adjust the path for output CSV file
df_results.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")

"""***Emotions***"""

import pandas as pd
from openai import OpenAI
import os
import time

# Initialize OpenAI client with your API key
client = OpenAI(
    api_key='.......',
)

def analyse_user_prompt(text):
    time.sleep(1)
    system_msg = """You are an expert in discourse analysis and psychology.


    Each of reviews have been posted by current and former employees. Based on the analysis, identify what emotions of employee are presented by the review posted by the employee.
    One review may present more than one type of contexts
    1.	Joy - High
    2.	Joy - Medium
    3.	Joy - Low
    4.	Trust - High
    5.	Trust - Medium
    6.	Trust - Low
    7.	Fear - High
    8.	Fear - Medium
    9.	Fear - Low
    10.	Surprise - High
    11.	Surprise - Medium
    12.	Surprise - Low
    13.	Sadness - High
    14.	Sadness - Medium
    15.	Sadness - Low
    16.	Anticipation - High
    17.	Anticipation - Medium
    18.	Anticipation - Low
    19.	Anger - High
    20.	Anger - Medium
    21.	Anger - Low
    22.	Disgust - High
    23.	Disgust - Medium
    24.	Disgust - Low


    Example of an analysis for your reference.
    Review -{I had a terrible experience when handling customers, my manager hardly trained me on how to do much of anything so when I had an actual customer I would stutter and stagger. She didn't bother to help me when handling customers so when I made a mistake she wrote it down and sent a report that I was disobeying when in reality I was just confused on some small things here and there. I did not enjoy my time working under that employer at the time.}
    Answer - Joy - Low, Trust - Low, Fear - Medium, Sadness - High, Anger - High, Disgust - Medium

    You can extract the answer to the CSV file as one column for ID and another column for the answer
          """

    messages=[
        {
          "role": "system",
          "content": system_msg
        },
        {
          "role": "user",
          "content": text
        }]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content

# Read reviews from CSV file
input_csv_path = 'File_Name.csv'  # Adjust the path to your input CSV file
df_reviews = pd.read_csv(input_csv_path)

# Analyze each review
emotions = []
for review in df_reviews['review']:
    result = analyse_user_prompt(review)
    emotions.append(result)  # Collect results

# Create a DataFrame for the results
df_results = pd.DataFrame({
    'review': df_reviews['review'],
    ' emotions':  emotions
})

# Save results to a new CSV file
output_csv_path = ' emotions.csv'  # Adjust the path for output CSV file
df_results.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")

"""***Personality Traits***"""

import pandas as pd
from openai import OpenAI
import os
import time

# Initialize OpenAI client with your API key
client = OpenAI(
    api_key='.......',
)

def analyse_user_prompt(text):
    time.sleep(1)
    system_msg = """You are an expert in discourse analysis and psychology.


    Each of reviews have been posted by current and former employees. Based on the analysis, identify what traits of employee are presented by the review posted by the employee.
    One review may present more than one type of contexts
    1.	Openness - High
    2.	Openness - Medium
    3.	Openness - Low
    4.	Conscientiousness - High
    5.	Conscientiousness - Medium
    6.	Conscientiousness - Low
    7.	Extraversion - High
    8.	Extraversion - Medium
    9.	Extraversion - Low
    10.	Agreeableness - High
    11.	Agreeableness - Medium
    12.	Agreeableness - Low
    13.	Neuroticism - High
    14.	Neuroticism - Medium
    15.	Neuroticism - Low



    Example of an analysis for your reference.
    Review -{I had a terrible experience when handling customers, my manager hardly trained me on how to do much of anything so when I had an actual customer I would stutter and stagger. She didn't bother to help me when handling customers so when I made a mistake she wrote it down and sent a report that I was disobeying when in reality I was just confused on some small things here and there. I did not enjoy my time working under that employer at the time.}
    Answer - Openness - Low, Conscientiousness - Medium, Extraversion - Low, Agreeableness - Medium, Neuroticism - High

    You can extract the answer to the CSV file as one column for ID and another column for the answer
          """

    messages=[
        {
          "role": "system",
          "content": system_msg
        },
        {
          "role": "user",
          "content": text
        }]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content

# Read reviews from CSV file
input_csv_path = 'File_Name.csv'  # Adjust the path to your input CSV file
df_reviews = pd.read_csv(input_csv_path)

# Analyze each review
traits = []
for review in df_reviews['review']:
    result = analyse_user_prompt(review)
    traits.append(result)  # Collect results

# Create a DataFrame for the results
df_results = pd.DataFrame({
    'review': df_reviews['review'],
    ' traits':  traits
})

# Save results to a new CSV file
output_csv_path = ' traits.csv'  # Adjust the path for output CSV file
df_results.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
