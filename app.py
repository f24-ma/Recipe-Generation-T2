import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------------------
# Load Model from Hugging Face
# -------------------------------------------------------------
MODEL_PATH = "flax-community/gpt-2-recipe-generator"  # Use a real recipe model

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

# -------------------------------------------------------------
# Function to clean text
# -------------------------------------------------------------
def clean_text(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    cleaned = []
    seen = set()
    for line in lines:
        lower = line.lower()
        if lower not in seen and len(line.split()) > 2:
            if not (line.endswith(":") or line.startswith("Recipe:")):
                cleaned.append(line)
                seen.add(lower)
    return "\n".join(cleaned)

# -------------------------------------------------------------
# Streamlit Interface
# -------------------------------------------------------------
st.title("Recipe Generator")
st.write("Type a dish name or list of ingredients to generate a clear and realistic recipe.")

text = st.text_area("Enter dish or ingredients (e.g., 'chocolate cake', 'eggs, milk, flour'): ")

if st.button("Generate Recipe"):
    if not text.strip():
        st.warning("Please enter something first.")
    else:
        prompt = (
            f"Write a detailed and realistic cooking recipe for {text}. "
            f"Include sections for:\n"
            f"Recipe Name: {text}\n"
            f"Ingredients:\n- (list of ingredients)\n\n"
            f"Instructions:\n1. (numbered steps)\n"
            f"Make it professional, readable, and non-repetitive."
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=2.5,
                no_repeat_ngram_size=6,
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
