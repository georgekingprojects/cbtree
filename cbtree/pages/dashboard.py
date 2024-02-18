"""The dashboard page."""
from cbtree.templates import template
from ..classify import predict_cognitive_distortions

from openai import OpenAI
import requests
import json

import reflex as rx

class FormInputState(rx.State):
    form_data: dict = {}
    show_results: bool = False

    @staticmethod
    def get_distortion_description(distortion):
        """Get description for a given distortion."""
        distortion_descriptions = {
            "All-or-nothing thinking": "This is a kind of polarized thinking. This involves looking at a situation as either black or white or thinking that there are only two possible outcomes to a situation. An example of such thinking is, 'If I am not a complete success at my job; then I am a total failure.'",
            "Overgeneralization": "When major conclusions are drawn based on limited information, or some large group is said to have same behavior or property. For example: 'one nurse was rude to me, this means all medical staff must be rude.' or 'last time I was in the pool I almost drowned, I am a terrible swimmer and should not go into the water again.'",
            "Mental filter": "A person engaging in filter (or 'mental filtering) takes the negative details and magnifies those details while filtering out all positive aspects of a situation. This means: focusing on negatives and ignoring the positives. If signs of either of these are present, then it is marked as mental filter.",
            "Should statements": "Should statements ('I should pick up after myself more') appear as a list of ironclad rules about how a person should behave, this could be about the speaker themselves or other. It is NOT necessary that the word 'should' or it’s synonyms (ought to, must etc.) be present in the statements containing this distortion. For example: consider the statement – 'I don’t have ups and downs like teenagers are supposed to; everything just seems kind of flat with a few dips', this suggests that the person believes that a teenager should behave in a certain way and they are not conforming to that pattern, this makes it a should statement cognitive distortion.",
            "Labeling": "Labeling is a cognitive distortion in which people reduce themselves or other people to a single characteristic or descriptor, like 'I am a failure.' This can also be a positive descriptor such as 'we were perfect'. Note that the tense in these does not always have to be present tense.",
            "Personalization": "Personalizing or taking up the blame for a situation which is not directly related to the speaker. This could also be assigning the blame to someone who was not responsible for the situation that in reality involved many factors and was out of your/the person’s control. The first entry in the sample is a good example for this.",
            "Magnification": "Blowing things way out of proportion. For example: 'If I don’t pass this test, I would never be successful in my career'. The impact of the situation here is magnified. You exaggerate the importance of your problems and shortcomings, or you minimize the importance of your desirable qualities. Not to be confused with mental filter, you can think of it only as maximizing the importance or impact of a certain thing.",
            "Emotional Reasoning": "Basically, this distortion can be summed up as - 'If I feel that way, it must be true.' Whatever a person is feeling is believed to be true automatically and unconditionally. One of the most common representation of this is some variation of – ‘I feel like a failure so I must be a failure’. It does not always have to be about the speaker themselves, 'I feel like he is not being honest with me, he must be hiding something' is also an example of emotional reasoning.",
            "Mind Reading": "Any evidence of the speaker suspecting what others are thinking or what are the motivations behind their actions. Statements like 'they won’t understand', 'they dislike me' suggest mind reading distortion. However, 'she said she dislikes me' is not a distortion, but 'I think she dislikes me since she ignored me' is again mind reading distortion (since it is based on assumption that you know why someone behaved in a certain way).",
            "Fortune-telling": "As the name suggests, this distortion is about expecting things to happen a certain way, or assuming that thing will go badly. Counterintuitively, this distortion does not always have future tense, for example: 'I was afraid of job interviews so I decided to start my own thing' here the person is speculating that the interview will go badly and they will not get the job and that is why they decided to start their own business. Despite the tense being past, the error in thinking is still fortune-telling."
        }
        return distortion_descriptions.get(distortion, "")

    def handle_submit(self, form_data: dict):
        """Handle the form submit."""
        # Call predict_cognitive_distortions function with form text and the full path to the CSV file
        predictions = predict_cognitive_distortions(form_data['input'])
        # Update form_data with predictions
        self.form_data = {**form_data, **predictions}  # Merge form_data with predictions
        
        # Check if the highest probability distortion is over 0.2
        highest_probability = max(predictions.values())
        if highest_probability > 0.2:
            # Find the distortion with the highest probability
            likely_distortion = max(predictions, key=predictions.get)
            # Set distortion description for the likely distortion
            distortion_desc = self.get_distortion_description(likely_distortion)
        else:
            likely_distortion = "No distortion"
            distortion_desc = ""

        # Add likely_distortion and distortion_desc to form_data
        self.form_data['likely_distortion'] = likely_distortion
        self.form_data['distortion_desc'] = distortion_desc

        client = OpenAI(api_key="5a13e04381894ee10904a767636ce96b200d2312c2b09d8ada7ad208e1f4a2ce",
            base_url='https://api.together.xyz',
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a cognitive behavioral therapist. Your patient is about to tell you about a situation, which you should respond by mentioning cognitive distortions they may have face, {likely_distortion} specifically.",
                },
                {
                    "role": "user",
                    "content": f"{self.form_data['input']}",
                }
            ],
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_tokens=128
        )

        self.form_data["together"] = chat_completion.choices[0].message.content

        url = 'https://d28f0f02-846c-4edb-a2b9-49600797af9b.monsterapi.ai/generate'
        headers = {
            'accept': 'application/json',
            'Authorization': 'Bearer 44d2e009-2f29-4457-afd1-6a46595a5664',
            'Content-Type': 'application/json'
        }
        data = {
            "input_variables": {
                "prompt": f"What cognitive distortion does the following represent?: {self.form_data['input']}"
            },
            "stream": False,
            "n": 1,
            "temperature": 0,
            "max_tokens": 128
        }

        response = requests.post(url, headers=headers, json=data, verify=False)
        response_data = response.json()
        #text_list = response_data['text']
        # Sample string
        #data_string = "This is a sample\nstring with [some] data."

        # Replace all '\n' with an empty string
        response_data = response_data.replace('\n', '')

        # Find the index of the first '[' and ']'
        start_index = response_data.find('[')
        end_index = response_data.find(']')

        # Extract substring between '[' and ']', not inclusive
        if start_index != -1 and end_index != -1:
            extracted_data = response_data[start_index + 1:end_index]
        else:
            extracted_data = ""
        self.form_data["monster"] = extracted_data

        self.show_results = True


@template(route="/cbtree", title="Try CBTree")
def dashboard() -> rx.Component:
    """The dashboard page.

    Returns:
        The UI for the dashboard page.
    """
    return rx.chakra.vstack(
        rx.chakra.heading("CBTree", font_size="3em"),
        rx.chakra.text("Identify and challenge negative thought patterns using CBT techniques!"),
        rx.text("What's on your mind?"),
        rx.form.root(
            rx.vstack(
                rx.input(
                    name="input",
                    default_value="search",
                    placeholder="Input text here...",
                    #type="password",
                    required=True,
                ),
                rx.button("Submit", type="submit"),
                width="100%",
                align="center",
            ),
            on_submit=FormInputState.handle_submit,
            reset_on_submit=True,
            width="100%",
            align="center",
        ),
        rx.divider(width="100%"),
        rx.cond(
            FormInputState.show_results,
            rx.vstack(
                rx.heading("Results"),
                #rx.text(FormInputState.form_data.to_string()),
                rx.text(f"Most likely distortion: {FormInputState.form_data['likely_distortion']}"),
                rx.text(FormInputState.form_data['distortion_desc']),
                rx.heading("Together.AI Mixtral-8x7B-Instruct-v0.1 Response"),
                rx.text(FormInputState.form_data['together']),
                rx.heading("MonsterAPI Mixtral Fine-Tuned Response"),
                rx.text(FormInputState.form_data['monster']),
            ),
            rx.text("")
        ),
        width="100%",
        align="center"
    )
