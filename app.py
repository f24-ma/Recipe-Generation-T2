import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------------------
# Load Model (Public, works without authentication)
# -------------------------------------------------------------
MODEL_PATH = "distilgpt2"  #  Public model that runs on Streamlit Cloud

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -------------------------------------------------------------
# Clean text output
# -------------------------------------------------------------
def clean_text(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    cleaned, seen = [], set()
    for line in lines:
        low = line.lower()
        if low not in seen and len(line.split()) > 2:
            if not line.endswith(":") and not line.startswith("Recipe:"):
                cleaned.append(line)
                seen.add(low)
    return "\n".join(cleaned)

# -------------------------------------------------------------
# Streamlit Interface
# -------------------------------------------------------------
st.title(" Recipe Generator")
st.write("Enter a dish name or list of ingredients to generate a clear and realistic recipe.")

text = st.text_area("Enter dish or ingredients (e.g. 'pizza', 'eggs, flour, milk'): ")

if st.button("Generate Recipe"):
    if not text.strip():
        st.warning(" Please enter something first.")
    else:
        prompt = (
            f"Write a detailed, clear, and realistic cooking recipe for {text}. "
            f"Include sections for:\nIngredients:\n- (list items)\n\nInstructions:\n1. (numbered steps)\n"
            f"Keep it professional, readable, and non-repetitive."
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=220,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=2.5,
                no_repeat_ngram_size=5,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
        recipe = recipe.replace(prompt, "").strip()
        recipe = clean_text(recipe)
        recipe = recipe.replace("Ingredients", "\n\nIngredients").replace("Instructions", "\n\nInstructions")

        st.subheader("Generated Recipe:")
        st.text(recipe)
