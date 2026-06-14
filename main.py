col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Tell Me About the Resume"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": f"Summarize this resume in 4 bullet points:\n{resume_text[:1500]}"}]
                )
                st.session_state["output"] = response.choices[0].message.content

    with col2:
        if st.button("How Can I Improvise my Skills"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": f"Based on this resume and job description, suggest 4 skills the candidate should improve:\n\nJD: {jd[:800]}\n\nResume: {resume_text[:800]}"}]
                )
                st.session_state["output"] = response.choices[0].message.content

    with col3:
        if st.button("Percentage Match"):
            st.session_state["output"] = f"Match Score: {score}%"
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": f"Explain in 3 bullet points why this resume matches {score}% with the job description:\n\nJD: {jd[:800]}\n\nResume: {resume_text[:800]}"}]
                )
                st.session_state["output"] += "\n\n" + response.choices[0].message.content

    if "output" in st.session_state:
        st.divider()
        st.write(st.session_state["output"])
