# FAQ Caching

import streamlit as st
from database import FAQ

def get_cached_faq(db_session, question):
    faq = db_session.query(FAQ).filter(FAQ.question == question).first()
    if faq:
        return faq.answer
    return None

def cache_faq(db_session, question, answer):
    faq = FAQ(question=question, answer=answer)
    db_session.add(faq)
    try:
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        st.error(f"Failed to cache FAQ: {e}")
