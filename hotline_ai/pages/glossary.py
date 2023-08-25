import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

add_page_title()

st.markdown( 
    """
    ### <span style="color:red"> List of context </span>
    """
,unsafe_allow_html=True)

with st.expander("Temperature"):
    st.markdown("""          
    - Higher values will make the output more random, while lower values will make it more focused and deterministic.
    
    - Think of it as a chaos dial. If you turn up the temperature, you'll get more random and unexpected responses. If you turn it down, the responses will be more predictable and focused.
    """)

with st.expander("Top P"):
    st.markdown("""
    - An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    
    - This is like setting a rule that the AI can only choose from the best possible options. If you set top_p to 0.1, it's like telling the AI, You can only pick from the top 10% of your 'best guesses'.
    """)
 
with st.expander("Top K"):
    st.markdown("""
    - Introduces random sampling for generated tokens by randomly selecting the next token from the k most likely options.
    
    - This one is similar to top_p but with a fixed number. If top_k is set to 50, it's like telling the AI, You have 50 guesses. Choose the best one.
    """)

with st.expander("Presence Penalty"):
    st.markdown("""
    - Higher values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
                
    - This is like a tool to help the AI talk about more diverse topics. If you increase this, the AI will be less likely to mention things it has already talked about and more likely to bring up new topics.""")
 
with st.expander("Frequency Penalty"):
    st.markdown("""
    - Higher values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    
    - This one helps to prevent the AI from repeating itself. The higher this value, the less likely the AI is to use the same phrases over and over. It's like encouraging the AI to be more original in its responses.
    """)

with st.expander("Differences between top_p and top_k "):
    st.markdown("""
    - Think of top_k and top_p as two different ways of limiting the AI's options when it's trying to decide what to say next. They're like rules for how it should choose the next word in its response.

    - **Top_k:** is like giving the AI a specific number of options to pick from. If you set top_k to 10, then the AI will only consider the 10 most likely words it might say next. It's like asking a group of 10 experts what they think the next word should be, and then randomly picking one of their suggestions.

    - **Top_p:** is a bit different. Instead of setting a specific number of options, you're setting a percentage that captures a certain amount of likelihood. So if you set top_p to 0.9 (or 90%), the AI will consider as many words as it takes to add up to 90% of the total likelihood. That might be 5 words, or it might be 50, depending on the situation. It's like asking enough experts until you've covered 90% of the expertise in a field, and then picking randomly from their suggestions.

    - In both cases, the AI is making a random selection from a limited set of options, but the way those options are chosen is different. Both have their uses, depending on what kind of randomness and creativity you want in the AI's responses.
    """)

with st.expander("Differences between frequency_penalty and presence_penalty"):
    st.markdown("""
    - **Frequency_penalty:** Think of this as the AI's aversion to repeating itself too much. If you set this value high, the AI will try not to use the same words or phrases that it's already used a lot in the current conversation. It's like someone who hates telling the same story twice, so they always try to come up with new anecdotes or examples.

    - **Presence_penalty:** This one is about introducing new topics. If you set this value high, the AI will be encouraged to bring up things it hasn't mentioned yet in the current conversation. It's like someone who loves to change the subject a lot, always bringing up new topics to keep the conversation interesting.

    - So in a nutshell, frequency_penalty is about avoiding repetition of words and phrases, while presence_penalty is about encouraging diversity in topics.
    """)

with st.expander("Recommended good settings to generate texts for creative writing"):
    st.markdown("""
    - **Temperature:** You can set this to a moderate or higher value, around 0.7 to 1.0. This will allow the AI to produce more diverse and creative responses.

    - **Top_p:** You can set this to a relatively high value, around 0.9. This means the AI is allowed to consider a wide range of possible next words, allowing for more creativity.

    - **Top_k:** This parameter can be set around 40 to 50. It's a balance to ensure the AI doesn't go too off the rails with its predictions while still maintaining some level of creativity.

    - **Presence_penalty:** Set this to a low value if your creative writing needs a consistent theme or higher if you want more diverse ideas. Perhaps start with a value of 0.1 and adjust based on the results.

    - **Frequency_penalty:** This can be set to a moderate value, around 0.5. This will help the AI to not repeat the same phrases too frequently, which is important in creative writing.
    """)

with st.expander("Recommended good settings to generate scientific and accurate information"):
    st.markdown("""
    - **Temperature:** Set this lower, around 0.3 to 0.5. A lower temperature makes the output more deterministic and focused, which can be helpful for generating more precise and reliable responses.

    - **Top_p:** You can set this to a relatively high value, perhaps around 0.9. This would allow the AI to consider a broad range of the most probable responses, which can help to ensure accuracy.

    - **Top_k:** A lower value, like 20 to 30, might be a good starting point. This restricts the AI to picking from a smaller set of highly probable next words, which can help to maintain coherence and accuracy.

    - **Presence_penalty:** A low value, maybe around 0.1, should work well. This setting encourages the AI to stick to the topics it has already brought up, which can be helpful for maintaining focus in a technical or scientific discussion.
    
    - **Frequency_penalty:** Also set this to a lower value, around 0.2 to 0.4. This encourages the AI to reuse important terminology and phrases, which can be important for maintaining accuracy in a scientific context.
    
    - **Remember**, these are just suggestions and might not be perfect for every scenario. It could take some trial and error to find the best settings for your particular needs. Also, while these settings can help to promote accuracy, always remember to verify any important information generated by the AI using reliable sources.
    
    - **Check this** [topK-topP-PP-FP](https://www.reddit.com/r/LocalLLaMA/comments/157djvv/confused_about_temperature_top_k_top_p_repetition/)
    """)
# if genre == 'Comedy':
#     st.write('You selected comedy.')
# else:
#     st.write("You didn\'t select comedy.")