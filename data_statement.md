# DoDo Data Statement

## Curation Rationale

The purpose of the DoDo dataset is to train, evaluate, and refine language models for classification tasks related to understanding online conversations directed at footballers and MPs.

## Language Variety

Due to the UK-centric domains this dataset concerns (men's and women's UK football leagues, and UK MPs), all tweets are in English.

## Speaker Demographics

All entries are collected from Twitter and therefore generally represent the demographics of the platform. The sample is skewed towards those engaging in community discussion of the two domains on the platform (sports and politics).

## Annotator Demographics
The two domains used differing annotator pools. For the MPs data, we made use of a company offering annotation services that recruited 23 annotators to work for 5 weeks in early 2023. The annotators were screened from an initial pool of 36 annotators who took a test consisting of 36 difficult gold-standard questions (containing examples of all four class labels). The annotators had constant access to both a core team member from the service provider and from the core research team.

Fifteen annotators self-identified as women, and eight as men. The annotators were sent an optional survey to provide further information on their demographics. Out of 23 annotators, 21 responded to the survey. By age, 12 annotators were between 18-29 years old, eight were between 30-39 years old, and one was over 50 years old. In terms of completed education level, three annotators had high school degrees, eight annotators had undergraduate degrees, six annotators had postgraduate taught degrees, and four annotators had postgraduate research degrees. The majority of annotators were British (17), and other nationalities included Indian, Swedish, and United States. Twelve annotators identified as White, with one identifying as White Other and one identifying as White Arab. Other ethnicities included Black Caribbean (1), Indian (1), Indian British Asian (1), and Jewish (1). Most annotators identified as heterosexual (14), with other annotators identifying as bisexual (3), gay (1), and pansexual (1). Two chose not to disclose their sexuality. The majority stated that English was their native language (16), and four stated they were not native but fluent in the language. One chose not to disclose whether they were native English speakers or not. The majority of annotators disclosed that they spend 1-2 hours per day on social media (12). Four annotators stated that they spent, on average, less than 1 hour on social media per day (but more than 10 minutes), and five stated they spend more than 2 hours per day on social media. Some of the annotators reported having themselves been targeted by online abuse (9), with 11 reporting `never' and one preferring not to say.

The datasets for footballers were annotated separately using a crowdsourcing platform. Due to this, we have significantly less detail on the demographics of the users. The fb-m dataset was annotated by 3,375 crowdworkers from 41 countries. The fb-w dataset was annotated by 3,513 crowdworkers from 48 countries. The annotators for both datasets were primarily from Venezuela (56\% and 64\% respectively) and the United States (29\% and 18\% respectively).   


## Speech Situation

The data consists of short-form written textual entries from social media (Twitter). These were presented and interpreted in isolation for labelling, i.e., not in a comment thread and without user/network or any additional information.

## Text Characteristics

The genre of texts is a mix of abusive, critical, positive, and neutral social media entries (tweets). Of the 12,000 total entries annotated, 8,304 were labelled positive, 7,360 were labelled neutral, 7,878 were labelled critical, and 4,458 were labelled abusive.

