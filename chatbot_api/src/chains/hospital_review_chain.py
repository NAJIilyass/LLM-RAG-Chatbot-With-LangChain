import os

from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")



HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding= hf,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="reviews",
    node_label="Review",
    text_node_properties=[
        "physician_name",
        "patient_name",
        "text",
        "hospital_name",
    ],
    embedding_node_property="embedding",
)

review_template = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer,
say you don't know.
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=review_template
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

reviews_vector_chain = RetrievalQA.from_chain_type(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
        huggingfacehub_api_token=HUGGINGFACE_API_KEY, 
        temperature=0.5, model_kwargs={"max_length": 500},
    ),
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)
reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt

query = """What have patients said about hospital efficiency?
...         Mention details from specific reviews."""

response = reviews_vector_chain.invoke(query)
print(response.get("result"))
