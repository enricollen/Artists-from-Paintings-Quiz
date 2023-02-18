# Artists From Paintings Quiz
A simple python flask application that implements a quiz game against an AI model (CNN classifier) where the user needs to guess the name of the author of an art picture.

Each game consists in a 10 paintings guessing rounds, where the user is asked to make a choice between 4 different painters (out of 11 total, at each game authors changes). At each guessing, the result of the player increments by 1 in case of correct answer, else remains the same (no penalties), and so does the score of AI model that in background computes its prediction at each new image.

At the end of a game, the score of the player and the score of the AI model are reported, and you can choose to play again the game with new images and authors.

The model that represents the opposing player is a VGG16 pre-trained ConvNet with a test accuracy of nearly 80% on a similar dataset, so prepare yourself, it won't be easy :P

Below i attach some screenshots from the game:
![game](https://user-images.githubusercontent.com/55366018/219872599-1597362a-611f-4ac4-94aa-77690d8e9eb3.png)
![results](https://user-images.githubusercontent.com/55366018/219872603-5b0c1fff-5f18-44a2-85be-9ee99e8fba95.png)
