import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
MODEL_PATH = "MuqadusAmsha/gpt2-recipe-generator"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

def clean_text(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    cleaned, seen = [], set()
    for line in lines:
        low = line.lower()
        if low not in seen and len(line.split()) > 2:
            if not line.endswith(":"):
                cleaned.append(line)
                seen.add(low)
    return "\n".join(cleaned)

st.title("Recipe Generator")
st.write("Type a dish name or list of ingredients to generate a clear and realistic recipe.")

text = st.text_area("Enter dish or ingredients:")

if st.button("Generate Recipe"):
    if not text.strip():
        st.warning("Please enter something first.")
    else:
        prompt = (
            f"Write a clear, realistic cooking recipe for {text}. "
            f"Include:\nIngredients list and numbered Instructions. Avoid repetition."
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=3.0,
                no_repeat_ngram_size=5,
                do_sample=True,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )

        recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
        recipe = recipe.replace(prompt, "").strip()
        recipe = clean_text(recipe)
        recipe = recipe.replace("Ingredients", "\n\nIngredients").replace("Instructions", "\n\nInstructions")

        st.subheader("Generated Recipe:")
        st.text(recipe)

