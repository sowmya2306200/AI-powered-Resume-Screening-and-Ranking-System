import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_file):
    text = extract_text(pdf_file)
    return text

def rank_resumes(job_description, resumes):
    # Combine job description and resumes
    documents = [job_description] + resumes

    # Vectorize the text
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    cosine_matrix = cosine_similarity(vectors)
    
    # Get similarity scores
    similarity_scores = cosine_matrix[0][1:]
    return similarity_scores

def main():
    st.title("AI-Powered Resume Screening & Ranking System")

    # Input for job description
    job_description = st.text_area("Job Description", "")
    
    # File upload for resumes
    uploaded_files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=["txt", "pdf"])
    
    # Process uploaded resumes
    if uploaded_files and job_description:
        resumes = []
        resume_names = []
        for file in uploaded_files:
            resume_names.append(file.name)  # Store the resume filename
            if file.type == "application/pdf":
                resumes.append(extract_text_from_pdf(file))
            else:
                resumes.append(file.read().decode("utf-8"))
        
        # Rank resumes
        scores = rank_resumes(job_description, resumes)
        
        # Display rankings with names
        st.write("**Resume Rankings:**")
        ranked_resumes = sorted(zip(resume_names, scores), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(ranked_resumes):
            st.write(f"**Rank {i+1}: {name} (Score: {score:.2f})**")

if __name__ == "__main__":
    main()
