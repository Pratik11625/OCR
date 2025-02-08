


import streamlit as st
import easyocr
from PIL import Image, ImageChops
from io import BytesIO
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParserfrom 

st.set_page_config(layout="wide", page_title="Image OCR & Comparison Tool")

st.title("🖼️ Image OCR & Comparison Tool")

uploaded_files = st.file_uploader("Upload two images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

def perform_ocr(image):
    reader = easyocr.Reader(['en'])
     # Convert PIL image to bytes
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    text = reader.readtext(img_bytes, detail=0)
    return "\n".join(text)

def compare_images(img1, img2):
    img2 = img2.resize(img1.size)
    diff = ImageChops.difference(img1, img2)
    return diff if diff.getbbox() else None

if len(uploaded_files) == 2:
    img1 = Image.open(uploaded_files[0]).convert("RGB")
    img2 = Image.open(uploaded_files[1]).convert("RGB")
    
    option = st.radio("Choose an option:", [
        "1️⃣ Perform OCR on an Image",
        "2️⃣ Compare Two Images",
        "3️⃣ Q&A on Extracted Text"
    ])
    
    if option == "1️⃣ Perform OCR on an Image":
        img_choice = st.radio("Select Image for OCR:", ["Image 1", "Image 2"])
        chosen_img = img1 if img_choice == "Image 1" else img2
        st.image(chosen_img, caption=f"{img_choice}", use_column_width=True)
        
        extracted_text = perform_ocr(chosen_img)
        st.text_area("Extracted Text:", extracted_text, height=200)
        st.session_state["extracted_text"] = extracted_text  # Save for Q&A
    
    elif option == "2️⃣ Compare Two Images":
        diff_img = compare_images(img1, img2)
        if diff_img:
            st.image(diff_img, caption="Differences Highlighted", use_column_width=True)
        else:
            st.success("No differences detected!")
    
    elif option == "3️⃣ Q&A on Extracted Text":
        if "extracted_text" in st.session_state:
            st.text_area("Extracted OCR Text:", st.session_state["extracted_text"], height=200)
            
            user_question = st.text_input("Ask a question about the text:")
            if user_question:
                template = PromptTemplate(input_variables=["question", "text"],
                                          template="Given the text: \"{text}\", answer the question: \"{question}\".")
                output = StrOutputParser()
                llm = OllamaLLM(model="qwen2.5:7b")
                chain =  template | llm | output
                response = chain.invoke(input={"question": user_question, "text": st.session_state["extracted_text"]})
                st.write("### Response:")
                st.write(response)
        else:
            st.warning("Please perform OCR first to extract text!")
else:
    st.info("Please upload exactly two images.")


















# from PIL import Image, ImageChops
# import streamlit as st

# # Open images

# img_1 = st.file_uploader('upload the img1', type=['jpg','png','jpeg'])
# img_2 = st.file_uploader('upload the img2', type=['jpg','png','jpeg'])

# img1 = Image.open(img_1).convert("RGB")
# img2 = Image.open(img_2).convert("RGB")

# # Resize img2 to match img1
# img2 = img2.resize(img1.size)

# # Compute difference
# diff = ImageChops.difference(img1, img2)

# if diff.getbbox():  # If there's a difference
#     diff.show()
# else:
#     print("No differences found")
