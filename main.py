import streamlit as st
import tensorflow as tf
import numpy as np
from itertools import zip_longest
import streamlit as st
from streamlit_chat import message
import googleapiclient.discovery
import streamlit as st
import googleapiclient.discovery

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


api_key = "AIzaSyDgwVYiML9g9_5YbIYgKzRxVZ632nIr4PU"  # Replace with your API key
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

if 'predicted_dish_recipe' not in st.session_state:
    st.session_state['predicted_dish_recipe'] = ""

dish=""
# Initialize the ChatOpenAI model
chat = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo", 
    max_tokens=2000
)

recipe_model = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo", 
    max_tokens=2000
)

def generate_recipe(dish):
    Instruction = [SystemMessage(
        content = f"""Your name is Digital Chef. You are a Cooking Expert for {dish} which is traditional Indian cuisine, here to guide and assist people with their cooking and recipe-related questions and concerns. Please provide accurate and helpful information and always maintain a polite and professional tone.
          1. The your output consist of Four heading 
            1st should be Ingredients to cook/make/prepare the {dish},
            2nd should be Instruction to cook/make/prepare the {dish},
            3th should be other similar and reconmended dish like  the {dish}
        2. You must avoid discussing sensitive, offensive, or harmful content. Refrain from engaging in any form of discrimination, harassment, or inappropriate behavior.
        3. Be patient and considerate when responding to user queries and provide clear explanations.
        4. If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
        5. If the user asks about dish recipe other than the {dish} traditional Indian cuisine,say strickly NO and tell the user i have the knowledge about the traditional Indian cuisine.
        Remember, your primary goal is to assist and educate people in the field of cooking and recipe of traditional Indian cuisine.""" )]
            
    return Instruction   




def build_message_list(dish):
    """
    Build a list of messages including system, human, and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
            
             content = """Your name is Digital Chef. You are a Cooking Expert for traditional Indian cuisine, here to guide and assist people with their cooking and recipe-related questions and concerns. Please provide accurate and helpful information and always maintain a polite and professional tone.

                    1. Greet the user politely, ask for their name, and inquire about how you can assist them with cooking and recipe queries.
                    2. Provide informative and relevant responses to questions like, "What is the recipe for a user asked specific dish?" "How many ingredients are needed?" "Which ingredients are used to cook it?" "How much time is required?" etc.
                    3. Also, must provide the YouTube videos links related that tell the user how to make  the user entered specific dish?
                    4. Remember, Must provide the related youtube videos links.
                    5. You must avoid discussing sensitive, offensive, or harmful content. Refrain from engaging in any form of discrimination, harassment, or inappropriate behavior.
                    6. Be patient and considerate when responding to user queries and provide clear explanations.
                    7. If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
                    8. If the user asks about dish recipe other than the traditional Indian cuisine,say strickly NO and tell the user i have the knowledge about the traditional Indian cuisine.
                    9. Must provide the video of that recipe.
                    Remember, your primary goal is to assist and educate people in the field of cooking and recipe of traditional Indian cuisine.
                    Note : The output consist of three heading 1st is Ingredients and 2nd is Ingredients and 3rd is Related youtube videos links"""

    )]

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg))  # Add user messages
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg))  # Add AI messages

    return zipped_messages

def generate_response():
    """
    Generate AI response using the ChatOpenAI model.
    """
    # Build the list of messages
    zipped_messages = build_message_list(dish)

    # Generate response using the chat model
    # print(zipped_messages)
    ai_response = chat(zipped_messages)

    return ai_response.content

# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element


def Your_digital_chef():
    # Create a text input for user
    st.text_input('YOU: ', key='prompt_input', on_change=submit)
    if st.session_state.entered_prompt != "":
        # Get user query
        user_query = st.session_state.entered_prompt

        # Append user query to past queries
        st.session_state.past.append(user_query)

        # Generate response
        output = generate_response()

        # Append AI response to generated responses
        st.session_state.generated.append(output)

        # Display the chat history
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                # Display AI response
                message(st.session_state["generated"][i], key=str(i))
                # Display user message
                message(st.session_state['past'][i],
                        is_user=True, key=str(i) + '_user')


#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

#Main Page
if(app_mode=="Home"):   
    st.header("INDIAN CUISINE RECOGNITION SYSTEM")
    image_path = "bannar.png"
    st.image(image_path)
    st.header("Your Digital Chef")
    Your_digital_chef()

#About Project
elif(app_mode=="About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("Dishes -- adhirasam, aloo_gobi, aloo_matar, aloo_methi, aloo_shimla_mirch, aloo_tikki, anarsa, ariselu, bandar_laddu, basundi, bhatura, bhindi_masala, biryani, boondi, butter_chicken, chak_hao_kheer, cham_cham, chana_masala, chapati, chhena_kheeri, chicken_razala, chicken_tikka, chicken_tikka_masala, chikki, daal_baati_churma, daal_puri, dal_makhani, dal_tadka, dharwad_pedha, doodhpak, double_ka_meetha, dum_aloo, gajar_ka_halwa, gavvalu, ghevar, gulab_jamun, imarti, jalebi, kachori, kadai_paneer, kadhi_pakoda, kajjikaya, kakinada_khaja, kalakand, karela_bharta, kofta, kuzhi_paniyaram, lassi, ledikeni, litti_chokha, lyangcha, maach_jhol, makki_di_roti_sarson_da_saag, malapua, misi_roti, misti_doi, modak, mysore_pak, naan, navrattan_korma, palak_paneer, paneer_butter_masala, phirni, pithe, poha, poornalu, pootharekulu, qubani_ka_meetha, rabri, ras_malai, rasgulla, sandesh, shankarpali, sheer_korma, sheera, shrikhand, sohan_halwa, sohan_papdi, sutar_feni, unni_appam")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (4000 images - 50 images of each category)")

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=3,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is Predicting it's a {}".format(label[result_index]))
        dish=label[result_index]
        # --------------------Chat bot start from here --------------------
        Inst = generate_recipe(dish)
        recipe = recipe_model(Inst)
        st.info(recipe.content)
        

        # --------------------Fetching Related Video --------------------
        query = "how to cook the " + dish
        search_response = youtube.search().list(
            q=query,
            type="video",
            part="id,snippet",
            maxResults=5  # You can adjust the number of results
        ).execute()

        videos = []
        for item in search_response['items']:
            video_id = item['id']['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            videos.append(video_url)

        # Display embedded videos
        st.write("Top 5 Videos:")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["video 1", "video 2", "video 3","video 4","video 5"])

        with tab1:
            st.video(videos[0])

        with tab2:
            st.video(videos[1])
        
        with tab3:
            st.video(videos[2])

        with tab4:
            st.video(videos[3])
        
        with tab5:
            st.video(videos[4])

        


