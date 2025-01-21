import os
from dotenv import load_dotenv
import cv2
import easyocr
import time
import numpy as np
import streamlit as st
from PIL import Image
from langchain_ollama import OllamaLLM
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

# Load environment variables
load_dotenv()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Initialize the language model
llm = OllamaLLM(model="qwen2.5:7b")

# HF_TOKEN = os.getenv('HF_TOKEN')

# if HF_TOKEN:
#     repo_id = "google/gemma-1.1-7b-it"

#     llm = HuggingFaceEndpoint(
#         repo_id=repo_id,
#         max_length=512,
#         temperature=0.5,
#         huggingfacehub_api_token=HF_TOKEN,
        
#     )

# else:
#     st.error("HF_TOKEN api key is not connecting")



# Set up the Streamlit app configuration
st.set_page_config(page_title="Image-based Q&A System", page_icon=":mag_right:", layout="centered")

def query_text(question, context):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Given the following context:\n{context}\nAnswer the question: {question}"
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    answer = chain.invoke({"context": context, "question": question})
    return answer

def summarize_content(context):
    prompt = PromptTemplate(
        input_variables=["context"],
        template="Summarize the following content:\n{context}"
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({"context": context})

# Application Title and Description
st.title("🖼️ Image-based Q&A System with OCR")
st.write("Upload an image containing text, and ask questions about the extracted content or summarize it.")

# File uploader for images
uploaded_image = st.file_uploader("Upload an image containing text", type=["jpg", "png", "jpeg"])
if uploaded_image:
    start = time.process_time()
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to OpenCV format and perform OCR with EasyOCR
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    easy_ocr_result = reader.readtext(image_cv, detail=1, paragraph=False, rotation_info=[90, 180 ,270, 360])
    

    # Extract recognized texts and display them
    recognized_texts = [result[1] for result in easy_ocr_result]
    extracted_text = "\n".join(recognized_texts)

    # Save to a text file
    with open('extracted.txt', 'w') as file:
        file.write(extracted_text)
    

    # Record response time
    end = time.process_time()
    st.info(f"Response time: {end - start:.2f} seconds")

    if st.button("Show OCR Image"):
        start = time.process_time()
        annotated_image = image_cv.copy()
        
        for result in easy_ocr_result:
            coord, text, confidence = result

            # text = " ".join([c if ord(c) < 128 else " " for c in text]).strip()
            
            # Draw bounding boxes and labels on the image
            (topleft, topright, bottomright, bottomleft) = coord
            tx, ty = int(topleft[0]), int(topleft[1])
            bx, by = int(bottomright[0]), int(bottomright[1])
            cv2.rectangle(annotated_image, (tx, ty), (bx, by), (0, 255, 0), 2)
            cv2.putText(annotated_image, text, (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 123, 143), 2)

        # Convert BGR to RGB for display and show the annotated image
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_image_rgb, caption="Annotated Image with EasyOCR Results")
        # Record response time
        end = time.process_time()
        st.info(f"Response time: {end - start:.2f} seconds")

    if extracted_text:
        extracted_text = open('extracted.txt').read()
        st.text_area("Extracted Text", extracted_text, height=200)

        # Button: Ask Question on Extracted Text
         
       
       
        # Button: Summarize the Content
        if st.button("Summarize Content"):
            start = time.process_time()
            summary = summarize_content(extracted_text)
            if summary:
                st.success("Summary:")
                st.write(summary)
            else:
                st.warning("Could not generate a summary.")
            
            # Record response time
            end = time.process_time()
            st.info(f"Response time: {end - start:.2f} seconds") 
            
        if st.button("Ask Question"):
                start = time.process_time()
                question = st.text_input("Enter your question about the extracted text:")
                # try: 
                #     if question:
                answer = query_text(question, extracted_text)
                st.success(f"Answer: {answer}")
                #     else:
                #         st.error("Please enter a question.")
                # except Exception as e :
                #     st.error(f" error: {e}")

                # Record response time
                print(answer)
                end = time.process_time()
                st.info(f"Response time: {end - start:.2f} seconds")

    else:
        st.warning("No text was extracted from the image.")
