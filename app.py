import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------------------------------------
# Use a public recipe-trained model (no login required)
# -------------------------------------------------------------
MODEL_PATH = "flax-community/t5-recipe-generation"  # Public model

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -------------------------------------------------------------
# Clean and format the generated text
# -------------------------------------------------------------
def clean_text(text):
    text = text.strip()
    text = text.replace("Ingredients:", "\n\n**Ingredients:**\n")
    text = text.replace("instructions:", "\n\n**Instructions:**\n")
    text = text.replace("Directions:", "\n\n**Instructions:**\n")
    return text.strip()

# -------------------------------------------------------------
# Streamlit interface
# -------------------------------------------------------------
st.title(" Recipe Generator")
st.write("Enter a dish name or ingredients and get a professional, realistic recipe!")

text = st.text_area("Enter dish or ingredients:")

if st.button("Generate Recipe"):
    if not text.strip():
        st.warning(" Please enter a dish name or ingredients first.")
    else:
        with st.spinner("Generating recipe... Please wait "):
            prompt = f"generate recipe: {text}"

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.8
            )

            recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
            recipe = clean_text(recipe)

        st.success("Recipe generated successfully!")
        st.markdown(recipe)
