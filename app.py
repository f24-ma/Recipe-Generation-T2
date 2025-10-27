import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your fine-tuned model
MODEL_PATH = "models/gpt2_recipe"  # or "your-username/gpt2-recipe-gen"

@st.cache_resource
def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model(MODEL_PATH)

st.set_page_config(page_title="Recipe Generator App")
st.title("Recipe Generator")
st.write("Generate a full recipe given a dish name or a list of ingredients using your fine-tuned GPT-2 model.")

prompt = st.text_area("Enter a dish name or ingredients:")

if st.button("Generate Recipe"):
    if not prompt.strip():
        st.warning("Please enter a dish name or ingredients.")
    else:
        input_text = f"Title: {prompt}\nIngredients:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=250,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Generated Recipe")
        st.text_area("Output:", generated, height=300)

st.markdown("---")
st.caption("Fine-tuned GPT-2 model for recipe generation — 24F-7812")
