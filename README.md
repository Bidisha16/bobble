# bobble
bobble
Q1. There are 60 data points which are randomly split into 3 classes of equal size. All partitions are equally likely. A and B are two data points among them. What is the probability that two data points A and B will end up in the same class?

ANSWER: A) 1/3

Q2. A report is stating that a person is suffering from COVID-19 though he is actually not affected by coronavirus. Which of the following case is correct

ANSWER: B) FALSE POSITIVE

Q3. Which of the following time series data can be declared as stationary?

ANSWER: A)

Q6. What are the effects of pooling operation on a CNN based model Statement1: Increase feature / dimensionality 
Statement2: Reduce feature / dimensionality 
Statement3: Performs down sampling operations 
Statement4: Performs up sampling operations

ANSWER: (D) Statement 2 and Statement 3 are correct

Q7. Role of activation function
 Statement1:Introduce non-linearity 
Statement2: Delivers output based on linear combinations of inputs Statement3: To learn complex patterns in data

ANSWER: (A) Statement 1 and Statement 3 are correct

Q8. Which of the following is a widely used and effective machine learning algorithm based on the idea of bagging?

ANSWER: (D) Random forest

Q9. How do you handle missing or corrupted data in a dataset?

ANSWER: (D) All of the above

Q10. Why is second order differencing in time series needed?

ANSWER: (C) Both A and B 
Q11. When performing regression or classification, which of the following is the correct way to preprocess the data?

ANSWER: (A) Normalize the data -> PCA -> training

Q12. Which of the following is an example of feature extraction?

ANSWER: (D) All of the above

Q13. Which of the following is true about Naive Bayes ?

ANSWER: (C) Both A and B

Q14. Which of the following statements about regularization is not correct?

ANSWER: (D) None of the above

Q15. How can you prevent a clustering algorithm from getting stuck in bad local optima?

ANSWER: (B) Use multiple radom initializations

Q16. In which of the following cases will K-means clustering fail to give good results? 1) Data points with outliers 2) Data points with different densities 3) Data points with nonconvex shapes

ANSWER: (C) 1,2 and 3

Q17. Which of the following is a reasonable way to select the number of principal components "k"?

ANSWER: (A) Choose k to be the smallest value so that at least 99% of the varinace is retained

Q18. You run gradient descent for 15 iterations with a=0.3 and compute J(theta) after each iteration. You find that the value of J(Theta) decreases quickly and then levels off. Based on this, which of the following conclusions seems most plausible?

ANSWER: (C) a=0.3 is an effective choice of learning rate

Q19. Consider the following neural network which takes two binary-valued inputs and outputs . Which of the following logical functions does it (approximately) compute?

Answer:  

Q24. What is padding in image processing

ANSWER: (C) The process of adding layers of zeros to an image

Part B: Fill in the blanks with correct option

Q1. Information gain leads to ……………………….. (increase/decrease) in Entropy?

ANSWER : DECREASE

Q2. A decision tree is a ………. (linear/nonlinear) algorithm for CLASSIFICATION (classificationn /regression) which works by trying to ……………………...(increase/decrease) entropy.

ANSWER : NON LINEAR, CLASSIFICATION, DECREASE

Q3. Logistic regression is used for……………………… (regression/classification)

ANSWER: CLASSIFICATION

Q4. K-Nearest Neighbours take more time for ……………………… (training/testing) less time for ……………………… (training/testing)

ANSWER: 

Perform the following in python and provide the code for the same:
1. Text lowercase
2. Change numerals to their counter names
3. Remove punctuations
4. Remove whitespaces
5. Remove stopwords
6. Stemming
7. Lemmatize
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled16.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "fUq7n77lWmiv",
        "outputId": "3f29d945-50d7-422f-f6b5-0ab07cf934af",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 136
        }
      },
      "source": [
        "import nltk\n",
        "nltk.download('stopwords')\n",
        "nltk.download('punkt')\n",
        "nltk.download('wordnet')"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n",
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Package punkt is already up-to-date!\n",
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/wordnet.zip.\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YC63pZfkXSr5",
        "outputId": "dd9add22-1d12-4e3a-9a06-d0aa6d08ea64",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        }
      },
      "source": [
        "\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.tokenize import word_tokenize\n",
        "from nltk.stem import PorterStemmer \n",
        "from nltk.stem import WordNetLemmatizer\n",
        "def remove(a):\n",
        "    return \"\".join(a.split())\n",
        "a=\"\"\"Steve was born in Tokyo, Japan in 1950. He moved to London with his parents when he\n",
        "was 5 years old. Steve started school there and his father began work at the hospital.\n",
        "His mother was a house wife and he had four brothers.\n",
        "He lived in England for 2 years then moved to Amman, Jordan where he lived there for\n",
        "10 years. Steve then moved to Cyprus to study at the Mediterranean University.\n",
        "Unfortunately, he did not succeed and returned to Jordan. His parents were very\n",
        "unhappy so he decided to try in America.\n",
        "He applied to many colleges and universities in the States and finally got some\n",
        "acceptance offers from them. He chose Wichita State University in Kansas. His major\n",
        "was Bio-medical Engineering. He stayed there for bout six months and then he moved\n",
        "again to a very small town called Greensboro to study in a small college.\"\"\"\n",
        "punctuations = '''!()-[]{};:'\"\\,<>./?@#$%^&*_~'''\n",
        "no_punct = \"\"\n",
        "for char in a:\n",
        "   if char not in punctuations:\n",
        "       no_punct = no_punct + char\n",
        "print(\"lowercase:\")\n",
        "print(a.lower())\n",
        "print(\"remove punctuation:\")\n",
        "print(no_punct)\n",
        "print(\"remove white spaces\")\n",
        "print(remove(a))                                                                                                                                                             \n",
        "text_tokens = word_tokenize(a)\n",
        "tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]\n",
        "print(\"without stopwords\")\n",
        "print(tokens_without_sw) \n",
        "print(\"stemming\")                                                                                                                                              \n",
        "ps = PorterStemmer()\n",
        "words = a.split()\n",
        "  \n",
        "for w in words: \n",
        "    print(w, \" : \", ps.stem(w))\n",
        "print(\"lemmetize\")\n",
        "lemmatizer = WordNetLemmatizer()\n",
        "for w in words: \n",
        "  print(lemmatizer.lemmatize(w))"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "lowercase:\n",
            "steve was born in tokyo, japan in 1950. he moved to london with his parents when he\n",
            "was 5 years old. steve started school there and his father began work at the hospital.\n",
            "his mother was a house wife and he had four brothers.\n",
            "he lived in england for 2 years then moved to amman, jordan where he lived there for\n",
            "10 years. steve then moved to cyprus to study at the mediterranean university.\n",
            "unfortunately, he did not succeed and returned to jordan. his parents were very\n",
            "unhappy so he decided to try in america.\n",
            "he applied to many colleges and universities in the states and finally got some\n",
            "acceptance offers from them. he chose wichita state university in kansas. his major\n",
            "was bio-medical engineering. he stayed there for bout six months and then he moved\n",
            "again to a very small town called greensboro to study in a small college.\n",
            "remove punctuation:\n",
            "Steve was born in Tokyo Japan in 1950 He moved to London with his parents when he\n",
            "was 5 years old Steve started school there and his father began work at the hospital\n",
            "His mother was a house wife and he had four brothers\n",
            "He lived in England for 2 years then moved to Amman Jordan where he lived there for\n",
            "10 years Steve then moved to Cyprus to study at the Mediterranean University\n",
            "Unfortunately he did not succeed and returned to Jordan His parents were very\n",
            "unhappy so he decided to try in America\n",
            "He applied to many colleges and universities in the States and finally got some\n",
            "acceptance offers from them He chose Wichita State University in Kansas His major\n",
            "was Biomedical Engineering He stayed there for bout six months and then he moved\n",
            "again to a very small town called Greensboro to study in a small college\n",
            "remove white spaces\n",
            "StevewasborninTokyo,Japanin1950.HemovedtoLondonwithhisparentswhenhewas5yearsold.Stevestartedschoolthereandhisfatherbeganworkatthehospital.Hismotherwasahousewifeandhehadfourbrothers.HelivedinEnglandfor2yearsthenmovedtoAmman,Jordanwherehelivedtherefor10years.StevethenmovedtoCyprustostudyattheMediterraneanUniversity.Unfortunately,hedidnotsucceedandreturnedtoJordan.HisparentswereveryunhappysohedecidedtotryinAmerica.HeappliedtomanycollegesanduniversitiesintheStatesandfinallygotsomeacceptanceoffersfromthem.HechoseWichitaStateUniversityinKansas.HismajorwasBio-medicalEngineering.HestayedthereforboutsixmonthsandthenhemovedagaintoaverysmalltowncalledGreensborotostudyinasmallcollege.\n",
            "without stopwords\n",
            "['Steve', 'born', 'Tokyo', ',', 'Japan', '1950', '.', 'He', 'moved', 'London', 'parents', '5', 'years', 'old', '.', 'Steve', 'started', 'school', 'father', 'began', 'work', 'hospital', '.', 'His', 'mother', 'house', 'wife', 'four', 'brothers', '.', 'He', 'lived', 'England', '2', 'years', 'moved', 'Amman', ',', 'Jordan', 'lived', '10', 'years', '.', 'Steve', 'moved', 'Cyprus', 'study', 'Mediterranean', 'University', '.', 'Unfortunately', ',', 'succeed', 'returned', 'Jordan', '.', 'His', 'parents', 'unhappy', 'decided', 'try', 'America', '.', 'He', 'applied', 'many', 'colleges', 'universities', 'States', 'finally', 'got', 'acceptance', 'offers', '.', 'He', 'chose', 'Wichita', 'State', 'University', 'Kansas', '.', 'His', 'major', 'Bio-medical', 'Engineering', '.', 'He', 'stayed', 'bout', 'six', 'months', 'moved', 'small', 'town', 'called', 'Greensboro', 'study', 'small', 'college', '.']\n",
            "stemming\n",
            "Steve  :  steve\n",
            "was  :  wa\n",
            "born  :  born\n",
            "in  :  in\n",
            "Tokyo,  :  tokyo,\n",
            "Japan  :  japan\n",
            "in  :  in\n",
            "1950.  :  1950.\n",
            "He  :  He\n",
            "moved  :  move\n",
            "to  :  to\n",
            "London  :  london\n",
            "with  :  with\n",
            "his  :  hi\n",
            "parents  :  parent\n",
            "when  :  when\n",
            "he  :  he\n",
            "was  :  wa\n",
            "5  :  5\n",
            "years  :  year\n",
            "old.  :  old.\n",
            "Steve  :  steve\n",
            "started  :  start\n",
            "school  :  school\n",
            "there  :  there\n",
            "and  :  and\n",
            "his  :  hi\n",
            "father  :  father\n",
            "began  :  began\n",
            "work  :  work\n",
            "at  :  at\n",
            "the  :  the\n",
            "hospital.  :  hospital.\n",
            "His  :  hi\n",
            "mother  :  mother\n",
            "was  :  wa\n",
            "a  :  a\n",
            "house  :  hous\n",
            "wife  :  wife\n",
            "and  :  and\n",
            "he  :  he\n",
            "had  :  had\n",
            "four  :  four\n",
            "brothers.  :  brothers.\n",
            "He  :  He\n",
            "lived  :  live\n",
            "in  :  in\n",
            "England  :  england\n",
            "for  :  for\n",
            "2  :  2\n",
            "years  :  year\n",
            "then  :  then\n",
            "moved  :  move\n",
            "to  :  to\n",
            "Amman,  :  amman,\n",
            "Jordan  :  jordan\n",
            "where  :  where\n",
            "he  :  he\n",
            "lived  :  live\n",
            "there  :  there\n",
            "for  :  for\n",
            "10  :  10\n",
            "years.  :  years.\n",
            "Steve  :  steve\n",
            "then  :  then\n",
            "moved  :  move\n",
            "to  :  to\n",
            "Cyprus  :  cypru\n",
            "to  :  to\n",
            "study  :  studi\n",
            "at  :  at\n",
            "the  :  the\n",
            "Mediterranean  :  mediterranean\n",
            "University.  :  university.\n",
            "Unfortunately,  :  unfortunately,\n",
            "he  :  he\n",
            "did  :  did\n",
            "not  :  not\n",
            "succeed  :  succeed\n",
            "and  :  and\n",
            "returned  :  return\n",
            "to  :  to\n",
            "Jordan.  :  jordan.\n",
            "His  :  hi\n",
            "parents  :  parent\n",
            "were  :  were\n",
            "very  :  veri\n",
            "unhappy  :  unhappi\n",
            "so  :  so\n",
            "he  :  he\n",
            "decided  :  decid\n",
            "to  :  to\n",
            "try  :  tri\n",
            "in  :  in\n",
            "America.  :  america.\n",
            "He  :  He\n",
            "applied  :  appli\n",
            "to  :  to\n",
            "many  :  mani\n",
            "colleges  :  colleg\n",
            "and  :  and\n",
            "universities  :  univers\n",
            "in  :  in\n",
            "the  :  the\n",
            "States  :  state\n",
            "and  :  and\n",
            "finally  :  final\n",
            "got  :  got\n",
            "some  :  some\n",
            "acceptance  :  accept\n",
            "offers  :  offer\n",
            "from  :  from\n",
            "them.  :  them.\n",
            "He  :  He\n",
            "chose  :  chose\n",
            "Wichita  :  wichita\n",
            "State  :  state\n",
            "University  :  univers\n",
            "in  :  in\n",
            "Kansas.  :  kansas.\n",
            "His  :  hi\n",
            "major  :  major\n",
            "was  :  wa\n",
            "Bio-medical  :  bio-med\n",
            "Engineering.  :  engineering.\n",
            "He  :  He\n",
            "stayed  :  stay\n",
            "there  :  there\n",
            "for  :  for\n",
            "bout  :  bout\n",
            "six  :  six\n",
            "months  :  month\n",
            "and  :  and\n",
            "then  :  then\n",
            "he  :  he\n",
            "moved  :  move\n",
            "again  :  again\n",
            "to  :  to\n",
            "a  :  a\n",
            "very  :  veri\n",
            "small  :  small\n",
            "town  :  town\n",
            "called  :  call\n",
            "Greensboro  :  greensboro\n",
            "to  :  to\n",
            "study  :  studi\n",
            "in  :  in\n",
            "a  :  a\n",
            "small  :  small\n",
            "college.  :  college.\n",
            "lemmetize\n",
            "Steve\n",
            "wa\n",
            "born\n",
            "in\n",
            "Tokyo,\n",
            "Japan\n",
            "in\n",
            "1950.\n",
            "He\n",
            "moved\n",
            "to\n",
            "London\n",
            "with\n",
            "his\n",
            "parent\n",
            "when\n",
            "he\n",
            "wa\n",
            "5\n",
            "year\n",
            "old.\n",
            "Steve\n",
            "started\n",
            "school\n",
            "there\n",
            "and\n",
            "his\n",
            "father\n",
            "began\n",
            "work\n",
            "at\n",
            "the\n",
            "hospital.\n",
            "His\n",
            "mother\n",
            "wa\n",
            "a\n",
            "house\n",
            "wife\n",
            "and\n",
            "he\n",
            "had\n",
            "four\n",
            "brothers.\n",
            "He\n",
            "lived\n",
            "in\n",
            "England\n",
            "for\n",
            "2\n",
            "year\n",
            "then\n",
            "moved\n",
            "to\n",
            "Amman,\n",
            "Jordan\n",
            "where\n",
            "he\n",
            "lived\n",
            "there\n",
            "for\n",
            "10\n",
            "years.\n",
            "Steve\n",
            "then\n",
            "moved\n",
            "to\n",
            "Cyprus\n",
            "to\n",
            "study\n",
            "at\n",
            "the\n",
            "Mediterranean\n",
            "University.\n",
            "Unfortunately,\n",
            "he\n",
            "did\n",
            "not\n",
            "succeed\n",
            "and\n",
            "returned\n",
            "to\n",
            "Jordan.\n",
            "His\n",
            "parent\n",
            "were\n",
            "very\n",
            "unhappy\n",
            "so\n",
            "he\n",
            "decided\n",
            "to\n",
            "try\n",
            "in\n",
            "America.\n",
            "He\n",
            "applied\n",
            "to\n",
            "many\n",
            "college\n",
            "and\n",
            "university\n",
            "in\n",
            "the\n",
            "States\n",
            "and\n",
            "finally\n",
            "got\n",
            "some\n",
            "acceptance\n",
            "offer\n",
            "from\n",
            "them.\n",
            "He\n",
            "chose\n",
            "Wichita\n",
            "State\n",
            "University\n",
            "in\n",
            "Kansas.\n",
            "His\n",
            "major\n",
            "wa\n",
            "Bio-medical\n",
            "Engineering.\n",
            "He\n",
            "stayed\n",
            "there\n",
            "for\n",
            "bout\n",
            "six\n",
            "month\n",
            "and\n",
            "then\n",
            "he\n",
            "moved\n",
            "again\n",
            "to\n",
            "a\n",
            "very\n",
            "small\n",
            "town\n",
            "called\n",
            "Greensboro\n",
            "to\n",
            "study\n",
            "in\n",
            "a\n",
            "small\n",
            "college.\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}

QUESTION
import numpy as np
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
np.random.seed(19680801)


fig, ax = plt.subplots()

resolution = 50  # the number of vertices
N = 3
x = np.random.rand(N)
y = np.random.rand(N)
radii = 0.1*np.random.rand(N)
patches = []
for x1, y1, r in zip(x, y, radii):
    circle = Circle((x1, y1), r)
   
    x = np.random.rand(N)
y = np.random.rand(N)
radii = 0.1*np.random.rand(N)
theta1 = 360.0*np.random.rand(N)
theta2 = 360.0*np.random.rand(N)
patches += [
    Wedge((.7, .8), .2, 0, 360, width=0.05),  
]


colors = 100 * np.random.rand(len(patches))
p = PatchCollection(patches, alpha=0.4)
p.set_array(colors)
ax.add_collection(p)
fig.colorbar(p, ax=ax)

plt.show()
