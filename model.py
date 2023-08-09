from langchain.document_loaders import PyPDFLoader, DirectoryLoader,TextLoader,WebBaseLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from torch import cuda, bfloat16
import transformers
from langchain.llms import HuggingFacePipeline
from transformers import StoppingCriteria, StoppingCriteriaList
DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """You are Humanoid robot name Kai. your task is provide information to people who will be talking to you.Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    # llm = CTransformers(
    #     model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
    #     model_type="llama",
    #     max_new_tokens = 512,
    #     temperature = 0.5
    # )
    model_id = 'meta-llama/Llama-2-7b-chat-hf'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    
    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    
    # begin initializing HF items, you need an access token
    hf_auth = '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'
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    
    # enable evaluation mode to allow model inference
    model.eval()
    print(f"Model loaded on {device}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id,use_auth_token=hf_auth)
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=256,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# #chainlit code
# @cl.on_chat_start
# async def start():
#     chain = qa_bot()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to Medical Bot. What is your query?"
#     await msg.update()

#     cl.user_session.set("chain", chain)

# @cl.on_message
# async def main(message):
#     chain = cl.user_session.get("chain") 
#     cb = cl.AsyncLangchainCallbackHandler(
#         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
#     )
#     cb.answer_reached = True
#     res = await chain.acall(message, callbacks=[cb])
#     answer = res["result"]
#     sources = res["source_documents"]

#     if sources:
#         answer += f"\nSources:" + str(sources)
#     else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

