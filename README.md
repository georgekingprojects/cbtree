# Welcome to CBTree!

Cognitive behavioral therapy (CBT) has been shown to be one of the most effective ways to treat depression.

Unfortunately, there is a shortage of CBT trained professionals, and there are often financial barriers to treatment.

CBTree is created as a way for people to experience something similar to CBT using machine learning, NLP, and LLMs.

I trained a classification model using a Kaggle dataset of patient statements annotated with cognitive distortions (negative thought patterns).

CBTree allows you to enter text and helps you identify and challenge these thought patterns. Additionally, LLMs are used to provide a conversational experience.

## Tech Stack

Python (Reflex, Scikit-learn, Pandas, etc)

Together.AI

MonsterAPI

### Cognitive distortions

1. All-or-nothing thinking
This is a kind of polarized thinking. This involves looking at a situation as either black or white or thinking that there are only two possible outcomes to a situation. An example of such thinking is, "If I am not a complete success at my job; then I am a total failure."

2. Overgeneralization
When major conclusions are drawn based on limited information, or some large group is said to have same behavior or property. For example: “one nurse was rude to me, this means all medical staff must be rude.” or “last time I was in the pool I almost drowned, I am a terrible swimmer and should not go into the water again”.

3. Mental filter
A person engaging in filter (or “mental filtering) takes the negative details and magnifies those details while filtering out all positive aspects of a situation. This means: focusing on negatives and ignoring the positives. If signs of either of these are present, then it is marked as mental filter.

4. Should statements
Should statements (“I should pick up after myself more”) appear as a list of ironclad rules about how a person should behave, this could be about the speaker themselves or other. It is NOT necessary that the word ‘should’ or it’s synonyms (ought to, must etc.) be present in the statements containing this distortion. For example: consider the statement – “I don’t have ups and downs like teenagers are supposed to; everything just seems kind of flat with a few dips”, this suggests that the person believes that a teenager should behave in a certain way and they are not conforming to that pattern, this makes it a should statement cognitive distortion.

5. Labeling
Labeling is a cognitive distortion in which people reduce themselves or other people to a single characteristic or descriptor, like “I am a failure.” This can also be a positive descriptor such as “we were perfect”. Note that the tense in these does not always have to be present tense.

6. Personalization
Personalizing or taking up the blame for a situation which is not directly related to the speaker. This could also be assigning the blame to someone who was not responsible for the situation that in reality involved many factors and was out of your/the person’s control. The first entry in the sample is a good example for this.

7. Magnification
Blowing things way out of proportion. For example: “If I don’t pass this test, I would never be successful in my career”. The impact of the situation here is magnified. You exaggerate the importance of your problems and shortcomings, or you minimize the importance of your desirable qualities. Not to be confused with mental filter, you can think of it only as maximizing the importance or impact of a certain thing.

8. Emotional Reasoning
Basically, this distortion can be summed up as - “If I feel that way, it must be true.” Whatever a person is feeling is believed to be true automatically and unconditionally. One of the most common representation of this is some variation of – ‘I feel like a failure so I must be a failure’. It does not always have to be about the speaker themselves, “I feel like he is not being honest with me, he must be hiding something” is also an example of emotional reasoning.

9. Mind Reading
Any evidence of the speaker suspecting what others are thinking or what are the motivations behind their actions. Statements like “they won’t understand”, “they dislike me” suggest mind reading distortion. However, “she said she dislikes me” is not a distortion, but “I think she dislikes me since she ignored me” is again mind reading distortion (since it is based on assumption that you know why someone behaved in a certain way).

10. Fortune-telling
As the name suggests, this distortion is about expecting things to happen a certain way, or assuming that thing will go badly. Counterintuitively, this distortion does not always have future tense, for example: “I was afraid of job interviews so I decided to start my own thing” here the person is speculating that the interview will go badly and they will not get the job and that is why they decided to start their own business. Despite the tense being past, the error in thinking is still fortune-telling.



### How to use

1. Install reflex via pip then do ```reflex run```